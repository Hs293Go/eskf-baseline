from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("eskf_baseline")

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "loam.yaml"]),
        description="Path to the loam_node parameter file",
    )
    imu_topic_arg = DeclareLaunchArgument(
        "imu_topic",
        default_value="/livox/imu",
        description="IMU topic (sensor_msgs/Imu)",
    )
    feature_topic_arg = DeclareLaunchArgument(
        "feature_topic",
        default_value="/arise_slam_mid360/feature_info",
        description="LaserFeature topic from arise_slam feature extractor",
    )

    loam_node = Node(
        package="eskf_baseline",
        executable="loam_node",
        name="loam_node",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
        remappings=[
            ("eskf/imu",          LaunchConfiguration("imu_topic")),
            ("loam/feature_info", LaunchConfiguration("feature_topic")),
        ],
    )

    return LaunchDescription([
        config_file_arg,
        imu_topic_arg,
        feature_topic_arg,
        loam_node,
    ])
