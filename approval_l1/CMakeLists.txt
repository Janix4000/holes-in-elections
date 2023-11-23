cmake_minimum_required(VERSION 3.12)
project(my_project)

# Source files
file(GLOB SOURCES "source/*.cpp")

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable debug flags
set(CMAKE_BUILD_TYPE Debug)

# Set the output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/out)

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options)

# Include Boost directories
include_directories(${Boost_INCLUDE_DIRS})

# Include directories for your project
include_directories("include")
include_directories("include/third-party")

# Create a library from your source files
add_library(my_lib STATIC ${SOURCES})

# Link against Boost libraries
target_link_libraries(my_lib ${Boost_LIBRARIES})

# Find all source files in the bindings directory
file(GLOB BINDING_SOURCES "bindings/*.cpp")

# Add the shared library
add_library(my_bindings SHARED ${BINDING_SOURCES})

# Link necessary libraries, for example, Boost libraries
target_link_libraries(my_bindings ${Boost_LIBRARIES})
target_link_libraries(my_bindings my_lib)

# Create executables for each script file
file(GLOB SCRIPTS "scripts/*.cpp")
foreach(script_file ${SCRIPTS})
    get_filename_component(script_name ${script_file} NAME_WE)
    add_executable(${script_name} ${script_file})
    target_link_libraries(${script_name} my_lib)
endforeach()
