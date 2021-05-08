cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_SUPPRESS_REGENERATION true)

# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

project(OpenCL_Vector_Addition)

find_package( OpenCL REQUIRED )

include_directories( ${OpenCL_INCLUDE_DIR} )
file(GLOB SOURCE_DIR
    "src/*.h"
    "src/*.hpp"
    "src/*.cpp"
)
add_executable(OpenCL_Vector_Addition ${SOURCE_DIR})
target_link_libraries(OpenCL_Vector_Addition ${OpenCL_LIBRARY})