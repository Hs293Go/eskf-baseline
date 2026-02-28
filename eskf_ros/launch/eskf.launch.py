from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    accelerometer_unit_is_g_arg = DeclareLaunchArgument(
        "accelerometer_unit_is_g",
        default_value="false",
        description="Set to true if IMU accelerometer data is in g instead of m/s^2",
    )

    pose_format_is_odometry_arg = DeclareLaunchArgument(
        "pose_format_is_odometry",
        default_value="false",
        description="Set to true if pose measurements are nav_msgs/Odometry",
    )

    input_topic_arg = DeclareLaunchArgument(
        "imu_topic", default_value="/imu/data", description="Input IMU topic"
    )

    meas_topic_arg = DeclareLaunchArgument(
        "meas_topic", default_value="/odom", description="Measurement (pose) topic"
    )

    # Launch configurations
    accelerometer_unit_is_g = LaunchConfiguration(
        "accelerometer_unit_is_g",
    )
    pose_format_is_odometry = LaunchConfiguration(
        "pose_format_is_odometry",
    )
    imu_topic = LaunchConfiguration("imu_topic")
    meas_topic = LaunchConfiguration("meas_topic")

    # Node definition
    eskf_node = Node(
        package="eskf_baseline",  # <-- replace with your actual package name
        executable="eskf_node",
        name="eskf_node",
        output="screen",
        parameters=[
            {
                "accelerometer_unit_is_g": accelerometer_unit_is_g,
                "pose_format_is_odometry": pose_format_is_odometry,
            }
        ],
        remappings=[
            ("eskf/imu", imu_topic),
            ("eskf/meas", meas_topic),
        ],
    )

    return LaunchDescription(
        [
            accelerometer_unit_is_g_arg,
            pose_format_is_odometry_arg,
            input_topic_arg,
            meas_topic_arg,
            eskf_node,
        ]
    )
