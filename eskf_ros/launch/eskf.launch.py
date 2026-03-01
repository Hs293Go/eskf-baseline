from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package share path
    pkg_share = FindPackageShare("eskf_baseline")

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "default.yaml"]),
        description="Path to the parameter config file",
    )

    input_topic_arg = DeclareLaunchArgument(
        "imu_topic", default_value="/imu/data", description="Input IMU topic"
    )

    meas_topic_arg = DeclareLaunchArgument(
        "meas_topic", default_value="/odom", description="Measurement (pose) topic"
    )

    # Launch configurations
    config_file = LaunchConfiguration("config_file")
    imu_topic = LaunchConfiguration("imu_topic")
    meas_topic = LaunchConfiguration("meas_topic")

    # Node definition
    eskf_node = Node(
        package="eskf_baseline",  # <-- replace with your actual package name
        executable="eskf_node",
        name="eskf_node",
        output="screen",
        parameters=[config_file],
        remappings=[
            ("eskf/imu", imu_topic),
            ("eskf/meas", meas_topic),
        ],
    )

    return LaunchDescription(
        [
            config_file_arg,
            input_topic_arg,
            meas_topic_arg,
            eskf_node,
        ]
    )
