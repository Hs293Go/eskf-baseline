project(eskf_core)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
      FIND_PACKAGE_ARGS 3.4 CONFIG)

set(BOOST_MATH_STANDALONE ON)
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/math/archive/refs/tags/boost-1.90.0.tar.gz
      SYSTEM FIND_PACKAGE_ARGS 1.74.0 CONFIG)
FetchContent_MakeAvailable(Eigen3 Boost)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(
    WARNING "Building in non-Release mode may lead to suboptimal performance.")
endif()

add_library(eskf_core_eskf_core)
target_sources(
  eskf_core_eskf_core
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         include
         FILES
         include/eskf_baseline/eskf_baseline.hpp
         include/eskf_baseline/outlier_classifier.hpp
         include/eskf_baseline/inertial_odometry_driver.hpp
         include/eskf_baseline/stats.hpp
         include/eskf_baseline/math.hpp
  PRIVATE src/stats.cpp src/outlier_classifier.cpp src/eskf.cpp)
target_compile_features(eskf_core_eskf_core PUBLIC cxx_std_20)
target_compile_options(eskf_core_eskf_core PUBLIC $<$<CONFIG:RELEASE>:-O3>)
target_link_libraries(
  eskf_core_eskf_core
  PUBLIC Eigen3::Eigen
  PRIVATE "$<COMPILE_ONLY:$<TARGET_NAME_IF_EXISTS:Boost::math>>"
          "$<COMPILE_ONLY:$<TARGET_NAME_IF_EXISTS:Boost::boost>>")
add_library(eskf_core::eskf_core ALIAS eskf_core_eskf_core)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
