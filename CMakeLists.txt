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
else()
  message(STATUS "MODULE_NAME not defined, skipping Python module build")
endif()
