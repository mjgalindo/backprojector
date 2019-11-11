#ifndef BACKPROJECT_CUDA
#define BACKPROJECT_CUDA

#include <inttypes.h>
#include <vector>
#include <array>

struct pointpair
{
	float cam_point[3], laser_point[3];
};

#ifndef MAKE_DLL_EXPORT
#define MAKE_DLL_EXPORT
#endif

MAKE_DLL_EXPORT
void call_cuda_backprojection(const float* transient_chunk,
                              uint32_t transient_size, uint32_t T,
                              const std::vector<pointpair> scanned_pairs,
                              const float* camera_position,
                              const float* laser_position,
                              float* voxel_volume,
                              const uint32_t* voxels_per_side,
                              const float* volume_zero_pos,
                              const float* voxel_inc,
                              float t0,
                              float deltaT);

MAKE_DLL_EXPORT
void call_cuda_octree_backprojection(const float* transient_chunk,
                              uint32_t transient_size, uint32_t T,
                              const std::vector<pointpair> scanned_pairs,
                              const float* camera_position,
                              const float* laser_position,
                              float* voxel_volume,
                              const uint32_t* voxels_per_side,
                              const float* volume_zero_pos,
                              const float* voxel_inc,
                              float t0,
                              float deltaT);


#endif