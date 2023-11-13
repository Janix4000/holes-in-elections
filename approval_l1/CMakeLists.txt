cmake_minimum_required(VERSION 3.12)
project(my_project)

# Source files
file(GLOB SOURCES "scripts/*.cpp")

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable debug flags
set(CMAKE_BUILD_TYPE Debug)

# Set the output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/out)


# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options)

# Include Boost directories
include_directories(${Boost_INCLUDE_DIRS})

include_directories("include/third-party")

# Link against Boost libraries
foreach(source_file ${SOURCES})
    get_filename_component(source_name ${source_file} NAME_WE)
    add_executable(${source_name} ${source_file})
    target_include_directories(${source_name} PUBLIC include)
    target_link_libraries(${source_name} ${Boost_LIBRARIES})
endforeach()