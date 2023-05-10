cmake_minimum_required(VERSION 3.12)
project(my_project)

# Source files
set(SOURCES
    src/brute.cpp
    src/greedy_dp.cpp
)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable debug flags
set(CMAKE_BUILD_TYPE Debug)

# Set the output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/out)

# Generate an executable for each source file
foreach(source_file ${SOURCES})
    get_filename_component(source_name ${source_file} NAME_WE)
    add_executable(${source_name} ${source_file})
    target_include_directories(${source_name} PUBLIC include)
endforeach()
