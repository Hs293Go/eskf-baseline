project(${SKBUILD_PROJECT_NAME})
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(${MODULE_NAME} src/eskf_baseline.cpp)
target_compile_definitions(${MODULE_NAME}
                           PRIVATE "PYBIND11_MODULE_NAME=${MODULE_NAME}")
target_link_libraries(${MODULE_NAME} PRIVATE pybind11::pybind11
                                             eskf_core::eskf_core)

install(TARGETS ${MODULE_NAME} DESTINATION .)
