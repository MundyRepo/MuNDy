cmake_minimum_required(VERSION 3.18)
project(sam_demo LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (either via package manager or submodule install of pybind11)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)  # falls back to non-CONFIG if available

# Build the Python extension module
pybind11_add_module(sam_demo module.cpp)

# Make sure the current directory is on the include path for headers
target_include_directories(sam_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# (Optional but nice): faster builds
if (MSVC)
  target_compile_options(sam_demo PRIVATE /permissive- /Zc:__cplusplus /W4)
else()
  target_compile_options(sam_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# On macOS you may want to ensure an rpath that lets Python load the module in-place
# set_target_properties(sam_demo PROPERTIES MACOSX_RPATH ON)
