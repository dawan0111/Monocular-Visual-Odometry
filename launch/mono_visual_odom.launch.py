from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import launch_ros
import os


def generate_launch_description():
    # Launch configuration variables
    node_name = LaunchConfiguration(
        "mono_visual_odometry_node", default="mono_visual_odometry_node"
    )

    # Node for publishing Eigen::Vector3d to PointCloud2
    mono_visual_odom_node = Node(
        package="mono_visual_odometry",
        executable="mono_visual_odometry_node",
        name=node_name,
        output="screen",
    )

    # Node for broadcasting a static transform for camera_optical_link
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "0",
            "0",
            "0",
            "-1.570796",
            "0",
            "-1.570796",
            "odom",
            "camera_optical_link",
        ],
        name="static_tf_publisher",
    )
    odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "map",
            "odom",
        ],
        name="static_tf_publisher",
    )

    return LaunchDescription(
        [
            mono_visual_odom_node,
            odom_tf,
            camera_tf,
            Node(
                package="rviz2",
                namespace="",
                executable="rviz2",
                name="rviz2",
            ),
        ]
    )
