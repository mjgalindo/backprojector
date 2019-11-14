#ifndef BACKPROJECT_CUDA
#define BACKPROJECT_CUDA

#include <inttypes.h>
#include <vector>
#include <array>
#include <complex>

struct ppd
{
    float camera_point[3], laser_point[3];
    float camera_wall, laser_wall;
};

#ifndef MAKE_DLL_EXPORT
#define MAKE_DLL_EXPORT
#endif

MAKE_DLL_EXPORT
void call_cuda_backprojection(const float* transient_chunk,
                              uint32_t transient_size, uint32_t T,
                              const std::vector<ppd> scanned_pairs,
                              float* voxel_volume,
                              const uint32_t* voxels_per_side,
                              const float* volume_zero_pos,
                              const float* voxel_inc,
                              float t0,
                              float deltaT);

MAKE_DLL_EXPORT
void call_cuda_octree_backprojection(const float* transient_chunk,
                                     uint32_t transient_size, uint32_t T,
                                     const std::vector<ppd> scanned_pairs,
                                     float* voxel_volume,
                                     const uint32_t* voxels_per_side,
                                     const float* volume_zero_pos,
                                     const float* voxel_inc,
                                     float t0,
                                     float deltaT);

MAKE_DLL_EXPORT
void call_cuda_complex_backprojection(const std::complex<float>* transient_chunk,
                                            uint32_t transient_size, uint32_t T,
                                            const std::vector<ppd> scanned_pairs,
                                            std::complex<float>* voxel_volume,
                                            const uint32_t* voxels_per_side,
                                            const float* volume_zero_pos,
                                            const float* voxel_inc,
                                            float t0, float deltaT);

#endif