include(FetchContent)
include(GoogleTest)
FetchContent_Declare(
  GTest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.tar.gz
      SYSTEM)
FetchContent_MakeAvailable(GTest)

add_executable(test_driver src/test_driver.cpp
                           src/test_inertial_odometry_integration.cpp)
target_link_libraries(test_driver PRIVATE GTest::gtest GTest::gtest_main
                                          GTest::gmock eskf_core::eskf_core)

gtest_discover_tests(test_driver)
