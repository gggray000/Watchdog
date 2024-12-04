#!/usr/bin/env python3
import math
import numpy
import rclpy
import threading
from collections import deque

from rclpy.node import Node
from rclpy.time import Time
from ros2topic.api import get_msg_class
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from driverless_messages.msg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from __main__ import Watchdog2

import rplogger.rplogger as rplog
from configuration.generalConfigReader import get_config, important_QoS


"""

TODO
1) Fix the countdown, so it would stop after 25s limit is reached.
2) Docstring or comment everything.

"""


def main():
    executor = MultiThreadedExecutor()
    executor.add_node(Watchdog2(config))
    executor.spin()


class Watchdog2(Node):

    def __init__(self, config):
        super().__init__("asm_watchdog")  # type: ignore
        self.logger: rplog.RPLogger = rplog.RPLogger("watchdog", rplog.LOG_LEVEL_INFO)
        self.watched_topics: list[WatchedTopic] = list()
        self.allStarted = False
        for key in config["watchdog"]["watches"]:
            self.watched_topics.append(WatchedTopic(self, self.logger, config["watchdog"]["watches"][key], key))

            self.start_timeout_timer = self.create_timer(
                config["watchdog"]["meta"]["start_timeout_check_interval_ms"] / 1000.0, self.check_start_timeouts
            )

    def check_start_timeouts(self):
        now = self.get_clock().now()
        timeout_limit = config["watchdog"]["meta"]["start_timeout_ms"]  # 25000
        for topic in self.watched_topics:
            topic.check_start_timeouts(now, timeout_limit)

        if not self.allStarted and all(topic.started_and_necessary() for topic in self.watched_topics):
            self.allStarted = True
            self.logger.log(rplog.LOG_LEVEL_INFO, "All nodes have been started.")
            self.start_timeout_timer.cancel()
            self.check_timeouts()

    def check_timeouts(self):
        if self.allStarted:
            for topic in self.watched_topics:
                thread = threading.Thread(target=topic.initialize_timer)
                thread.start()


class WatchedTopic:
    def __init__(self, node: Node, logger: rplog.RPLogger, sub_config: dict, key: str):
        self.node: Node = node
        self.logger: rplog.RPLogger = logger
        self.key: str = key
        self.sub_config = sub_config
        self.start_time: Time = node.get_clock().now()
        self.start_timeout_threshold = config["watchdog"]["meta"]["start_timeout_ms"]
        self.is_fatal: bool = sub_config["fatal"]
        self.list_maxlen = int(2000 / sub_config["expected_message_interval_ms"])
        self.last_seen_times_list = deque(maxlen=self.list_maxlen)
        self.has_published_fatal_error = False
        self.subscription = None
        self.phi: float = 0
        self.ifStarted = False

    def initialize_timer(self):
        self.timeout_timer = self.node.create_timer(
            self.sub_config["expected_message_interval_ms"] / 1000.0,
            self.check_phi,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def started_and_necessary(self):
        return self.ifStarted or not self.sub_config["necessary_on_startup"]

    def create_subscription(self, key):
        current_topics = [x[0] for x in self.node.get_topic_names_and_types()]
        if self.sub_config["topic"] not in current_topics:
            return
        callback_group = ReentrantCallbackGroup()
        msg_type = get_msg_class(self.node, self.sub_config["topic"])
        self.subscription = self.node.create_subscription(
            msg_type, self.sub_config["topic"], self.update_msg_list, important_QoS(), callback_group=callback_group
        )
        self.logger.log(rplog.LOG_LEVEL_INFO, f"Topic: {self.key} has been subscribed", rplog.LOG_COLOR_CYAN)

    def check_start_timeouts(self, now, timeout_limit):
        if self.subscription is None:
            self.create_subscription(self.key)

        if len(self.last_seen_times_list) == 0 and self.sub_config["necessary_on_startup"]:
            after_start_time = (now - self.start_time).nanoseconds / 1e9

            self.logger.log_throttle(
                1,
                rplog.LOG_LEVEL_INFO,
                "Time elapsed since AS startup without {} message: {:.2f}s. Stopping after {:.2f}s. Time remaining until fatal error is raised: {:.2f}s".format(
                    self.key,
                    after_start_time,
                    timeout_limit / 1000,
                    timeout_limit / 1000 - after_start_time,
                ),
                rplog.LOG_COLOR_YELLOW,
            )

            if after_start_time * 1000 > timeout_limit:
                self.timeout_triggered(self.key, after_start_time * 1000, True)

    def update_msg_list(self, _):
        self.last_seen_times_list.append(self.node.get_clock().now().nanoseconds / 1e9)
        self.ifStarted = True

    def calc_msg_interval(self, last_seen_times_list):
        timestamps_intervals = numpy.diff(last_seen_times_list)
        timestamps_stddev: float = numpy.std(timestamps_intervals)
        base_stddev_coefficient = 0.005
        min_stddev = self.sub_config["expected_message_interval_ms"] * base_stddev_coefficient
        valid_stddev = max(timestamps_stddev, min_stddev)
        return timestamps_intervals.mean(), valid_stddev

    def check_phi(self):
        # print(len(self.last_seen_times_list), "->", self.key)
        if len(self.last_seen_times_list) == self.list_maxlen:
            now = self.node.get_clock().now().nanoseconds / 1e9
            elapsed_time = now - self.last_seen_times_list[-1]
            timestamps_mean, valid_stddev = self.calc_msg_interval(self.last_seen_times_list)
            self.phi: float = self.calc_phi(elapsed_time, timestamps_mean, valid_stddev)
            self.logger.log(
                rplog.LOG_LEVEL_DEBUG,
                f"Topic: {self.key}, mean:{timestamps_mean: .4f}, stddev:{valid_stddev:.4f}, Current phi value: {self.phi:.10f}",
            )
            # phi > 3 indicates that the probability of wrong failure detection is 0.1%
            if self.phi > 3:
                self.timeout_triggered(self.key, elapsed_time, False)
        return self.phi

    def calc_phi(self, elapsed_time, timestamps_mean, timestamps_stddev):
        expected_mean = config["watchdog"]["watches"][self.key]["expected_message_interval_ms"] / 1000.0
        if timestamps_mean is None or timestamps_stddev is None:
            phi = 0
        else:
            p_later = 1 - 0.5 * (1 + math.erf((elapsed_time - expected_mean) / (timestamps_stddev * math.sqrt(2))))
            if p_later != 0:
                phi = abs(-math.log10(p_later))
            else:
                phi = 100
        return phi

    def timeout_triggered(self, key, elapsed_time, start_timeout):
        """
        This function handles the error sending if a timeout is triggered
        """
        if not self.has_published_fatal_error:
            if start_timeout:
                self.logger.log(
                    rplog.LOG_LEVEL_FATAL,
                    "Timeout for {} node after {}s since AS startup with limit of {}s".format(
                        key, elapsed_time, self.start_timeout_threshold / 1000
                    ),
                )
                # Start timeouts are always fatal!
                is_fatal = True
            else:
                self.logger.log(
                    rplog.LOG_LEVEL_FATAL if self.is_fatal else rplog.LOG_LEVEL_WARN,
                    "Timeout for {} node after {}ms. FATAL: {} ".format(key, elapsed_time * 1000, self.is_fatal),
                )
        self.has_published_fatal_error = True


if __name__ == "__main__":
    rclpy.init()
    config = get_config("asm")
    main()
    rclpy.shutdown()
