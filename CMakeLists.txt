if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "eskf_baseline_cpp")
endif()

cmake_minimum_required(VERSION 3.28.0)
project(${SKBUILD_PROJECT_NAME})

include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
      FIND_PACKAGE_ARGS 3.4 CONFIG)
FetchContent_MakeAvailable(Eigen3)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

add_library(${SKBUILD_PROJECT_NAME}_eskf_baseline INTERFACE)
target_sources(
  ${SKBUILD_PROJECT_NAME}_eskf_baseline
  INTERFACE FILE_SET
            HEADERS
            BASE_DIRS
            include
            FILES
            include/eskf_baseline/eskf_baseline.hpp
            include/eskf_baseline/math.hpp)
target_compile_features(${SKBUILD_PROJECT_NAME}_eskf_baseline
                        INTERFACE cxx_std_20)
target_link_libraries(${SKBUILD_PROJECT_NAME}_eskf_baseline
                      INTERFACE Eigen3::Eigen)
add_library(${SKBUILD_PROJECT_NAME}::eskf_baseline ALIAS
            ${SKBUILD_PROJECT_NAME}_eskf_baseline)

if(DEFINED MODULE_NAME)
  pybind11_add_module(${MODULE_NAME} module/eskf_baseline.cpp)
  target_compile_definitions(${MODULE_NAME}
                             PRIVATE "PYBIND11_MODULE_NAME=${MODULE_NAME}")
  target_link_libraries(
    ${MODULE_NAME} PRIVATE pybind11::pybind11
                           ${SKBUILD_PROJECT_NAME}::eskf_baseline)

  install(TARGETS ${MODULE_NAME} DESTINATION .)
else()
  message(STATUS "MODULE_NAME not defined, skipping Python module build")
endif()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${SKBUILD_PROJECT_NAME}::eskf_baseline)
