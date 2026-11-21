#!/usr/bin/env python3
"""Record ROS 2 Odometry topics as TUM trajectory files."""

import argparse
import os
import signal
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class TumRecorder(Node):
    def __init__(self, output_dir: Path, ground_truth_topic: str, estimate_topic: str):
        super().__init__("lio_sam_tum_recorder")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.gt_file = open(output_dir / "groundtruth.tum", "w", buffering=1)
        self.est_file = open(output_dir / "liosam.tum", "w", buffering=1)

        self.create_subscription(
            Odometry,
            ground_truth_topic,
            lambda msg: self.write_odom(self.gt_file, msg),
            100,
        )
        self.create_subscription(
            Odometry,
            estimate_topic,
            lambda msg: self.write_odom(self.est_file, msg),
            100,
        )

        self.get_logger().info(f"ground truth: {ground_truth_topic} -> {self.gt_file.name}")
        self.get_logger().info(f"estimate:     {estimate_topic} -> {self.est_file.name}")

    @staticmethod
    def write_odom(file_obj, msg: Odometry):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        file_obj.write(
            f"{stamp:.9f} {p.x:.9f} {p.y:.9f} {p.z:.9f} "
            f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"
        )

    def destroy_node(self):
        self.gt_file.close()
        self.est_file.close()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.expanduser("~/liosam_eval"))
    parser.add_argument("--ground-truth-topic", default="/odom")
    parser.add_argument("--estimate-topic", default="/lio_sam/mapping/odometry")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = TumRecorder(
        Path(args.output_dir),
        args.ground_truth_topic,
        args.estimate_topic,
    )

    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
