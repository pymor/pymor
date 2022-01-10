cmake_minimum_required(VERSION 3.1)

project(minimal-cpp-demo)

find_package(pybind11 CONFIG REQUIRED HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# Create an extension module
add_library(model MODULE model.cc)
target_link_libraries(model pybind11::module)
set_target_properties(model PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
set_target_properties(model PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")


