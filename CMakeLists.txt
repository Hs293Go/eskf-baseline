cmake_minimum_required(VERSION 3.28.0)
project(eskf_baseline)

if(DEFINED MODULE_NAME)
  set(BUILD_TESTING
      OFF
      CACHE BOOL "Disable building tests when building python module" FORCE)
endif()

add_subdirectory(eskf_core)

if(DEFINED MODULE_NAME AND SKBUILD_PROJECT_NAME)
  add_subdirectory(module)
  return()
else()
  message(STATUS "MODULE_NAME not defined, skipping Python module build")
endif()

find_package(ament_cmake)

if(ament_cmake_FOUND)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  message(STATUS "ament_cmake found, building ROS package")
  add_subdirectory(eskf_ros)

  ament_package()
endif()
