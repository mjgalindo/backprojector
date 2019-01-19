cmake_minimum_required(VERSION 3.11)

project(backprojector LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O3 -march=native")

add_executable(backprojector main.cpp)
add_library(cudabp SHARED backproject_cuda.cu)
add_library(cudabp_static STATIC backproject_cuda.cu)
include_directories(/opt/cuda/samples/common/inc)
set_target_properties(cudabp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cudabp PUBLIC cxx_std_14)

find_package(OpenMP REQUIRED)
add_definitions(-DXTENSOR_ENABLE_XSIMD)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
target_link_libraries(backprojector PRIVATE ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX cudabp_static)

install(TARGETS backprojector DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS cudabp DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES backproject.hpp backproject_cuda.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
