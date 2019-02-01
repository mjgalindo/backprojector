#ifndef BACKPROJECT_CUDA
#define BACKPROJECT_CUDA

#include <inttypes.h>
#include <vector>
#include <array>

const uint32_t MAX_THREADS_PER_BLOCK = 32;
const uint32_t MAX_BLOCKS_PER_KERNEL_RUN = 8;

struct pointpair
{
	float cam_point[3], laser_point[3];
};

void call_cuda_backprojection(const float* transient_chunk,
                              uint32_t transient_size, uint32_t T,
                              const std::vector<pointpair> scanned_pairs,
                              const float* camera_position,
                              const float* laser_position,
                              float* voxel_volume,
                              const uint32_t* voxels_per_side,
                              const float* volume_zero_pos,
                              const float* voxel_inc,
                              uint32_t t0,
                              float deltaT);

#endif