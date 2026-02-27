project(eskf_core)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
      FIND_PACKAGE_ARGS 3.4 CONFIG)
FetchContent_MakeAvailable(Eigen3)

add_library(eskf_core_eskf_core INTERFACE)
target_sources(
  eskf_core_eskf_core
  INTERFACE FILE_SET
            HEADERS
            BASE_DIRS
            include
            FILES
            include/eskf_baseline/eskf_baseline.hpp
            include/eskf_baseline/math.hpp)
target_compile_features(eskf_core_eskf_core INTERFACE cxx_std_20)
target_link_libraries(eskf_core_eskf_core INTERFACE Eigen3::Eigen)
add_library(eskf_core::eskf_core ALIAS eskf_core_eskf_core)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
