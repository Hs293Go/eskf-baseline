cmake_minimum_required(VERSION 3.22)
project(loam_baseline CXX)
#
# 1. Silence CMP0144: find_package uses upper-case <PACKAGENAME>_ROOT Setting
#   this to NEW tells CMake to use the ROOT variables without complaining.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# 1. Silence CMP0167: The FindBoost module is deprecated/removed Setting this to
#   OLD tells CMake to allow the legacy FindBoost PCL relies on.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED COMPONENTS common filters kdtree io)

# 1. Restore previous policy state so your own code remains modern
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
         include/loam_baseline/lidar_slam.hpp
         include/loam_baseline/irwgn.hpp
         include/loam_baseline/observation_models.hpp
         include/loam_baseline/feature_matcher.hpp
         include/loam_baseline/keypoint_extractor.hpp
         include/loam_baseline/LocalMap.h
         include/loam_baseline/flann/nanoflann.h
         include/loam_baseline/flann/octree.h
  PRIVATE src/lidar_slam.cpp src/keypoint_extractor.cpp)

target_compile_features(loam_baseline_core PUBLIC cxx_std_20)
target_compile_options(loam_baseline_core PUBLIC $<$<CONFIG:Release>:-O3>)

target_link_libraries(
  loam_baseline_core PUBLIC Eigen3::Eigen TBB::tbb
                            manifold_baseline::manifold_baseline)

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
