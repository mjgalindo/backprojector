project(dsreader)

cmake_minimum_required(VERSION 3.11)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O3 -march=native -ftree-vectorize")

add_executable(backprojector main.cpp)

add_definitions(-DXTENSOR_ENABLE_XSIMD)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)
target_link_libraries(backprojector ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)