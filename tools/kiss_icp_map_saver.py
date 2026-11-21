#!/usr/bin/env python3
"""Aggregate KISS-ICP keypoints into a global PCD map."""

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def write_binary_pcd(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA binary\n"
        )
        f.write(header.encode("ascii"))
        for p in points:
            f.write(struct.pack("<fff", float(p[0]), float(p[1]), float(p[2])))


class KissIcpMapSaver(Node):
    def __init__(self, output_path: Path, voxel_size: float):
        super().__init__("kiss_icp_map_saver")
        self.output_path = output_path
        self.voxel_size = voxel_size
        self.latest_pose = None
        self.voxels = {}
        self.cloud_count = 0

        self.create_subscription(Odometry, "/kiss/odometry", self.odom_callback, 100)
        self.create_subscription(PointCloud2, "/kiss/keypoints", self.cloud_callback, 10)
        self.get_logger().info(f"Saving KISS-ICP map to {output_path}")

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        rot = quat_to_rot(msg.pose.pose.orientation)
        trans = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.latest_pose = (rot, trans)

    def cloud_callback(self, msg: PointCloud2):
        if self.latest_pose is None:
            return
        rot, trans = self.latest_pose
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            local = np.array([point[0], point[1], point[2]], dtype=np.float64)
            global_point = rot @ local + trans
            key = tuple(np.floor(global_point / self.voxel_size).astype(np.int64))
            if key not in self.voxels:
                self.voxels[key] = global_point.astype(np.float32)
        self.cloud_count += 1
        if self.cloud_count % 100 == 0:
            self.get_logger().info(
                f"aggregated {self.cloud_count} clouds, {len(self.voxels)} voxels"
            )

    def save(self):
        points = list(self.voxels.values())
        write_binary_pcd(self.output_path, points)
        self.get_logger().info(f"wrote {len(points)} points to {self.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    args = parser.parse_args()

    rclpy.init()
    node = KissIcpMapSaver(args.output, args.voxel_size)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
