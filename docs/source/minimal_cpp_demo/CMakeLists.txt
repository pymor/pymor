cmake_minimum_required(VERSION 3.1)

set(Python3_EXECUTABLE /usr/local/bin/python3 CACHE PATH "")
project(minimal-cpp-demo)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

find_package(pybind11 CONFIG REQUIRED HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

# Create an extension module
add_library(model MODULE model.cc)
target_link_libraries(model pybind11::module)
set_target_properties(model PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
set_target_properties(model PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")


