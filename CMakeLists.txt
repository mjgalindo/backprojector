cmake_minimum_required(VERSION 3.11)

project(backprojector LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
#set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_COMPUTE_CAPABILITY 52 CACHE STRING "Cuda compute capability for the build")

set(XTL_INCLUDES xtl/include xtensor/include xtensor-blas/include)

add_executable(backprojector main.cpp)
target_compile_features(backprojector PUBLIC cxx_std_14)
add_library(cudabp SHARED backproject_cuda.cu)
add_library(cudabp_static STATIC backproject_cuda.cu)

set_target_properties(cudabp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(cudabp PUBLIC cxx_std_14)
target_compile_features(cudabp_static PUBLIC cxx_std_14)

# Compilation flags for most relevant cuda arquitectures up to turing (RTX)
set(CUDA_FLAGS $<$<CONFIG:Debug>:-G> -arch=sm_50 
        -gencode=arch=compute_50,code=sm_50 
        -gencode=arch=compute_52,code=sm_52 
        -gencode=arch=compute_60,code=sm_60 
        -gencode=arch=compute_61,code=sm_61 
        -gencode=arch=compute_70,code=sm_70 
        -gencode=arch=compute_75,code=sm_75 
        -gencode=arch=compute_75,code=compute_75)

target_compile_options(cudabp PRIVATE ${CUDA_FLAGS})
target_compile_options(cudabp_static PRIVATE ${CUDA_FLAGS})
target_include_directories(cudabp PRIVATE)
target_include_directories(cudabp_static PRIVATE)

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
target_include_directories(backprojector PRIVATE ${HDF5_INCLUDE_DIRS} ${XTL_INCLUDES})
if (WIN32)
    STRING(REGEX REPLACE "(.+)/.+.lib;.+" "\\1" HDF5_LIBRARY_DIR "${HDF5_LIBRARIES}")
    target_link_libraries(backprojector PRIVATE OpenMP::OpenMP_CXX cudabp_static 
        "${HDF5_LIBRARY_DIR}/libhdf5.lib;${HDF5_LIBRARY_DIR}/libhdf5_cpp.lib;${HDF5_LIBRARY_DIR}/libzlib.lib;${HDF5_LIBRARY_DIR}/libszip.lib")
else()
    target_link_libraries(backprojector PRIVATE OpenMP::OpenMP_CXX ${HDF5_LIBRARIES} cudabp_static)
endif(WIN32)

find_package (Python COMPONENTS Interpreter NumPy)
if (Python_NumPy_FOUND)
    message("-- Python bindings will be built")
    add_subdirectory(pybind11)
    pybind11_add_module(nlosbpy nlosbpy.cpp)
    target_compile_features(nlosbpy PUBLIC cxx_std_14)
    target_include_directories(nlosbpy PRIVATE ${HDF5_INCLUDE_DIRS} ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS} xtl/include xtensor/include xtensor-blas/include xtensor-python/include)
    target_link_libraries(nlosbpy PRIVATE ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX cudabp)
    install(TARGETS nlosbpy DESTINATION ${Python_SITELIB})
endif(Python_NumPy_FOUND)

install(TARGETS backprojector DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS cudabp cudabp_static DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES backproject.hpp backproject_cuda.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
