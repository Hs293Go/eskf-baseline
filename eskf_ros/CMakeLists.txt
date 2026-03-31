find_package(rclcpp REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(arise_slam_mid360_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

set(MSG_LIBRARIES std_msgs geometry_msgs sensor_msgs nav_msgs diagnostic_msgs)
foreach(MSG_LIB ${MSG_LIBRARIES})
  find_package(${MSG_LIB} REQUIRED)
  list(APPEND MSG_TARGETS ${MSG_LIB}::${MSG_LIB}__rosidl_typesupport_cpp)
endforeach()

# ---------------------------------------------------------------------------
# eskf_node — weakly-coupled baseline (pose topic in, ESKF out)
# ---------------------------------------------------------------------------
add_executable(eskf_node src/eskf_node.cpp)
target_link_libraries(
  eskf_node PRIVATE rclcpp::rclcpp eskf_core::eskf_core tf2_eigen::tf2_eigen
                    diagnostic_updater::diagnostic_updater ${MSG_TARGETS})

# ---------------------------------------------------------------------------
# loam_node — ESKF + LOAM (feature_info in, ESKF out)
#   Subscribes to arise_slam feature_info, runs loam_baseline::LidarSLAM,
#   feeds refined pose + GN covariance as R into the ESKF correction step.
# ---------------------------------------------------------------------------
add_executable(loam_node src/loam_node.cpp)
target_include_directories(loam_node PRIVATE ${pcl_conversions_INCLUDE_DIRS})
target_link_libraries(
  loam_node PRIVATE
    rclcpp::rclcpp
    eskf_core::eskf_core
    loam_baseline::core
    tf2_eigen::tf2_eigen
    diagnostic_updater::diagnostic_updater
    arise_slam_mid360_msgs::arise_slam_mid360_msgs__rosidl_typesupport_cpp
    ${MSG_TARGETS})

install(TARGETS eskf_node loam_node RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
