cmake_minimum_required(VERSION 3.11)

project(backprojector LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)

find_package(xtensor REQUIRED)

add_executable(backprojector main.cpp)
target_compile_features(backprojector PUBLIC cxx_std_17)
target_compile_options(backprojector PRIVATE -Wall)
add_library(cudabp SHARED backproject_cuda.cu)
add_library(cudabp_static STATIC backproject_cuda.cu)

set_target_properties(cudabp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cudabp PUBLIC cxx_std_14)
target_compile_features(cudabp_static PUBLIC cxx_std_14)

target_compile_options(cudabp PRIVATE $<$<CONFIG:Debug>:-G> --compiler-options -Wall,-fopenmp)
target_compile_options(cudabp_static PRIVATE $<$<CONFIG:Debug>:-G> --compiler-options -Wall,-fopenmp)

find_package(OpenMP REQUIRED)
target_include_directories(cudabp PUBLIC ${xtensor_INCLUDE_DIRS} ${xtl_INCLUDE_DIRS})
target_include_directories(cudabp_static PUBLIC ${xtensor_INCLUDE_DIRS} ${xtl_INCLUDE_DIRS})
target_link_libraries(cudabp PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(cudabp_static PUBLIC OpenMP::OpenMP_CXX)

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

target_link_libraries(backprojector PRIVATE ${HDF5_LIBRARIES} cudabp_static)

install(TARGETS backprojector DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS cudabp DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES backproject.hpp backproject_cuda.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
