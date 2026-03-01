find_package(rclcpp REQUIRED)
find_package(tf2_eigen REQUIRED)

set(MSG_LIBRARIES std_msgs geometry_msgs sensor_msgs nav_msgs)
foreach(MSG_LIB ${MSG_LIBRARIES})
  find_package(${MSG_LIB} REQUIRED)
  list(APPEND MSG_TARGETS ${MSG_LIB}::${MSG_LIB}__rosidl_typesupport_cpp)
endforeach()

add_executable(eskf_node src/eskf_node.cpp)
target_link_libraries(eskf_node PRIVATE rclcpp::rclcpp eskf_core::eskf_core
                                        tf2_eigen::tf2_eigen ${MSG_TARGETS})

install(TARGETS eskf_node RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
