cmake_minimum_required(VERSION 3.22)
project(loam_baseline CXX)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree io)
find_package(TBB REQUIRED)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(loam_baseline_core)

target_sources(
  loam_baseline_core
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         include
         FILES
         include/loam_baseline/lidarResiduals.hpp
         include/loam_baseline/LidarSlam.h
         include/loam_baseline/LocalMap.h
         include/loam_baseline/utils/EigenTypes.h
         include/loam_baseline/utils/Twist.h
         include/loam_baseline/utils/utility.h
         include/loam_baseline/utils/Utilities.h
         include/loam_baseline/flann/nanoflann.h
         include/loam_baseline/flann/octree.h
         include/loam_baseline/sensor_data/LidarPoint.h
  PRIVATE src/lidarResiduals.cpp src/LidarSlam.cpp)

target_compile_features(loam_baseline_core PUBLIC cxx_std_20)
target_compile_options(loam_baseline_core PUBLIC $<$<CONFIG:Release>:-O3>)

target_link_libraries(loam_baseline_core PUBLIC Eigen3::Eigen TBB::tbb)

# PCL does not always export modern targets on Ubuntu 22/24; use the variables
# as a fallback while still preferring imported targets.
if(TARGET pcl_common)
  target_link_libraries(loam_baseline_core PUBLIC pcl_common pcl_filters
                                                  pcl_kdtree pcl_io)
else()
  target_include_directories(loam_baseline_core PUBLIC ${PCL_INCLUDE_DIRS})
  target_link_libraries(loam_baseline_core PUBLIC ${PCL_LIBRARIES})
  target_compile_definitions(loam_baseline_core PUBLIC ${PCL_DEFINITIONS})
endif()

add_library(loam_baseline::core ALIAS loam_baseline_core)
