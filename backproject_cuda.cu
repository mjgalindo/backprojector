
#include <chrono>
#include <math.h>
#include <iostream>
#include <thread>
#include <chrono>

#ifdef _WIN32
#define MAKE_DLL_EXPORT __declspec(dllexport)
#endif
#include "backproject_cuda.hpp"
#include "tqdm.h"

__device__
float distance(const float *p1, const float *p2) 
{
	// Checking if either is a nullptr reduces performance dramatically!
	// if (p1 == nullptr || p2 == nullptr) return 0.0;
    float buff[3];
    for (int i = 0; i < 3; i++)
    {
        buff[i] = p1[i] - p2[i];
    }

    return norm3df(buff[0], buff[1], buff[2]);
}

__forceinline__ __device__
void compute_distance(const float * voxel_position,
					   const float *laser_pos, 
					   const float *camera_pos, 
					   const pointpair * pair,
					   float * distance_out)
{
	// From the laser to the wall
	float laser_wall_distance = distance(laser_pos, pair->laser_point);
	// From the wall to the current voxel
	float laser_point_voxel_distance = distance(pair->laser_point, voxel_position);
	// From the wall back to the camera
	float cam_wall_distance = distance(pair->cam_point, camera_pos);
	// From the object back to the wall
	float voxel_cam_point_distance = distance(voxel_position, pair->cam_point);
	
	*distance_out = laser_wall_distance + laser_point_voxel_distance + voxel_cam_point_distance + cam_wall_distance;
}

__forceinline__ __device__
void advance_block(const uint32_t *voxels_per_side, 
				   uint32_t * kernel_voxels, 
				   uint32_t & block_id,
				   uint32_t * xyz,
				   uint32_t & voxel_id)
{
	// First needs to find the x y z coordinates of the voxel
	block_id = (blockIdx.x * gridDim.y * gridDim.z +
				blockIdx.y * gridDim.z +
				blockIdx.z) * 3;
	for (uint32_t i = 0; i < 3; i++)
		xyz[i] = kernel_voxels[block_id+i]; // , kernel_voxels[block_id+1], kernel_voxels[block_id+2]};
	
	// Set the next voxel to be computed by this blockIdx in the next call.
	// We advance on the Z axis by the dimensions of the kernel, overflowing to the Y axis and 
	// then to the X axis. This would match row-major access to the 3D array.
	__syncthreads();
	if (threadIdx.x == 0)
	{
		uint32_t* next_xyz = &kernel_voxels[block_id];
		next_xyz[2] = xyz[2] + gridDim.z;
		if (next_xyz[2] >= voxels_per_side[2])
		{
			next_xyz[2] = next_xyz[2] % voxels_per_side[2];
			next_xyz[1] = next_xyz[1] + gridDim.y;
			if (next_xyz[1] >= voxels_per_side[1])
			{
				next_xyz[1] = next_xyz[1] % voxels_per_side[1];
				next_xyz[0] = next_xyz[0] + gridDim.x;
			}
		}
	}
	__syncthreads();
	voxel_id = xyz[0] * voxels_per_side[1] * voxels_per_side[2] +
			   xyz[1] * voxels_per_side[2] + 
			   xyz[2];
}


__global__
void cuda_backprojection_impl(float *transient_data,
                              uint32_t *T,
                              uint32_t *num_pairs,
                              pointpair *scanned_pairs,
                              float *camera_pos,
                              float *laser_pos,
                              float *voxel_volume,
                              float *volume_zero_pos,
                              float *voxel_inc,
                              float *t0,
                              float *deltaT,
							  uint32_t *voxels_per_side,
							  uint32_t *kernel_voxels)
{
	uint32_t block_id, voxel_id, xyz[3];
	advance_block(voxels_per_side, kernel_voxels, block_id, xyz, voxel_id);

	// If the block is outside the volume don't do anything
	if ((xyz[0] >= voxels_per_side[0]) | (xyz[1] >= voxels_per_side[1]) | (xyz[2] >= voxels_per_side[2]))
		return;
	
	extern __shared__ double local_array[];
	double& radiance_sum = local_array[threadIdx.x];
	radiance_sum = 0.0;

	// Don't run if the current voxel is not 0. This means the current block has already finished.
	//if (voxel_volume[voxel_id] == 0.0)
	{
		float voxel_position[] = {volume_zero_pos[0]+voxel_inc[0]*xyz[0],
								  volume_zero_pos[1]+voxel_inc[1]*xyz[1],
								  volume_zero_pos[2]+voxel_inc[2]*xyz[2]};

		for (uint32_t i = 0; i < *num_pairs / blockDim.x; i++)
		{
			uint32_t pair_index = i * blockDim.x + threadIdx.x;
			float total_distance;
			compute_distance(voxel_position, laser_pos, camera_pos, &scanned_pairs[pair_index], &total_distance);
			uint32_t time_index = round((total_distance - *t0) / *deltaT);
			uint32_t tdindex = pair_index * *T + time_index;

			radiance_sum += transient_data[tdindex]; // * distance_attenuation;
		}
	}
    __syncthreads();
	if (threadIdx.x == 0)
	{
		// Compute the reduction in a single thread and write it
		double total_radiance = 0.0;
		for (int i = 0; i < blockDim.x; i++) {
			total_radiance += local_array[i];
		}
		voxel_volume[voxel_id] = (float) total_radiance;
	}
    __syncthreads();
}

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
                              float deltaT)
{
    // Copy all the necessary information to the device
	/// float *transient_data,
	float *transient_chunk_gpu;
	cudaMalloc((void **)&transient_chunk_gpu, transient_size*sizeof(float));
    cudaMemcpy(transient_chunk_gpu, transient_chunk, transient_size*sizeof(float), cudaMemcpyHostToDevice);
	/// uint32_t *T,
	uint32_t *T_gpu;
	cudaMalloc((void **)&T_gpu, sizeof(uint32_t));
	cudaMemcpy(T_gpu, &T, sizeof(uint32_t), cudaMemcpyHostToDevice);
	/// uint32_t *num_pairs
    uint32_t *num_pairs_gpu;
    uint32_t num_pairs = scanned_pairs.size();
	cudaMalloc((void **)&num_pairs_gpu, sizeof(uint32_t));
	cudaMemcpy(num_pairs_gpu, &num_pairs, sizeof(uint32_t), cudaMemcpyHostToDevice);
	/// pointpair *scanned_pairs,
	pointpair *scanned_pairs_gpu;
	cudaMalloc((void **)&scanned_pairs_gpu, num_pairs * sizeof(pointpair));
	cudaMemcpy(scanned_pairs_gpu, scanned_pairs.data(), num_pairs * sizeof(pointpair), cudaMemcpyHostToDevice);
	/// float *camera_pos,
	float *camera_pos_gpu = nullptr;
	if (camera_position != nullptr)
	{
		cudaMalloc((void **)&camera_pos_gpu, 3 * sizeof(float));
		cudaMemcpy(camera_pos_gpu, camera_position, 3 * sizeof(float), cudaMemcpyHostToDevice);
	}
	/// float *laser_pos,
	float *laser_pos_gpu = nullptr;
	if (laser_position != nullptr)
	{
		cudaMalloc((void **)&laser_pos_gpu, 3 * sizeof(float));
		cudaMemcpy(laser_pos_gpu, laser_position, 3 * sizeof(float), cudaMemcpyHostToDevice);
	}
	/// float *voxel_volume,
	// Initialize voxel volume for the CPU
	float *voxel_volume_gpu;
	uint32_t nvoxels = voxels_per_side[0] * voxels_per_side[1] * voxels_per_side[2];
	cudaMalloc((void **)&voxel_volume_gpu, nvoxels * sizeof(float));
	cudaMemcpy(voxel_volume_gpu, voxel_volume, nvoxels * sizeof(float), cudaMemcpyHostToDevice);
	/// float *volume_zero_pos,
	float *volume_zero_pos_gpu;
	cudaMalloc((void **)&volume_zero_pos_gpu, 3 * sizeof(float));
	cudaMemcpy(volume_zero_pos_gpu, volume_zero_pos, 3 * sizeof(float), cudaMemcpyHostToDevice);
	/// float *voxel_inc,
	float *voxel_inc_gpu;
	cudaMalloc((void **)&voxel_inc_gpu, 3 * sizeof(float));
	cudaMemcpy(voxel_inc_gpu, voxel_inc, 3 * sizeof(float), cudaMemcpyHostToDevice);
	/// float *t0,
	float *t0_gpu;
	cudaMalloc((void **)&t0_gpu, sizeof(float));
	cudaMemcpy(t0_gpu, &t0, sizeof(float), cudaMemcpyHostToDevice);
	/// float *deltaT,
	float *deltaT_gpu;
	cudaMalloc((void **)&deltaT_gpu, sizeof(float));
	cudaMemcpy(deltaT_gpu, &deltaT, sizeof(float), cudaMemcpyHostToDevice);
	/// uint32_t *voxels_per_side,
	uint32_t *voxels_per_side_gpu;
	cudaMalloc((void **)&voxels_per_side_gpu, 3 * sizeof(uint32_t));
    cudaMemcpy(voxels_per_side_gpu, voxels_per_side, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	// uint32_t *kernel_voxels
	
	// Ideally, we would run {voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]} 
	// blocks, but windows doesn't like this, so we run many smaller kernels.
	
	{
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(1);
		}
	}

	int blockSize, minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_backprojection_impl, sizeof(double), 0); 
	{
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(1);
		}
	}

	// Limit blocksize to the number of pairs (or else backprojection will fail!)
	blockSize = std::min({(uint32_t) blockSize, num_pairs, 256u});

	// Force a smaller grid size to make each kernel run very short.
	minGridSize = 16;

	std::vector<uint32_t> kernel_voxels(minGridSize * minGridSize * minGridSize * 3);
	for (int x = 0; x < minGridSize; x++)
	for (int y = 0; y < minGridSize; y++)
	for (int z = 0; z < minGridSize; z++)
	{
		kernel_voxels[(x*minGridSize*minGridSize+y*minGridSize+z)*3 + 0] = x;
		kernel_voxels[(x*minGridSize*minGridSize+y*minGridSize+z)*3 + 1] = y;
		kernel_voxels[(x*minGridSize*minGridSize+y*minGridSize+z)*3 + 2] = z;
	}

	uint32_t *kernel_voxels_gpu;
	const uint32_t num_blocks_per_kernel_run = minGridSize*minGridSize*minGridSize;
	cudaMalloc((void **)&kernel_voxels_gpu, 3*num_blocks_per_kernel_run*sizeof(uint32_t));
	cudaMemcpy(kernel_voxels_gpu, kernel_voxels.data(), 3*num_blocks_per_kernel_run*sizeof(uint32_t), cudaMemcpyHostToDevice);

	dim3 xyz_blocks(minGridSize, minGridSize, minGridSize);
	dim3 threads_in_block(blockSize, 1, 1);
	uint32_t number_of_runs = std::ceil(std::max(std::max(voxels_per_side[0], voxels_per_side[1]), voxels_per_side[2]) / (float) minGridSize);
	number_of_runs = number_of_runs * number_of_runs * number_of_runs;

	std::cout << "Backprojecting on the GPU using the \"optimal\" configuration" << std::endl;
	std::cout << "# Blocks: " << xyz_blocks.x << ' ' << xyz_blocks.y << ' ' << xyz_blocks.z << std::endl;
	std::cout << "# Threads per block: " << threads_in_block.x << ' ' << threads_in_block.y << ' ' << threads_in_block.z << std::endl;
	std::cout << "# Kernel calls: " << number_of_runs << std::endl;
	
	auto start = std::chrono::steady_clock::now();
	tqdm bar;
	bar.set_theme_braille();
	for (uint32_t r = 0; r < number_of_runs; r++)
	{
		auto start = std::chrono::steady_clock::now();
		cuda_backprojection_impl<<<xyz_blocks, threads_in_block, blockSize*sizeof(double)>>>(transient_chunk_gpu,
			T_gpu,
			num_pairs_gpu,
			scanned_pairs_gpu,
			camera_pos_gpu,
			laser_pos_gpu,
			voxel_volume_gpu,
			volume_zero_pos_gpu,
			voxel_inc_gpu,
			t0_gpu,
			deltaT_gpu,
			voxels_per_side_gpu,
			kernel_voxels_gpu);
		cudaDeviceSynchronize();
		bar.progress(r, number_of_runs);
	}
	cudaDeviceSynchronize();
	bar.finish();
	auto end = std::chrono::steady_clock::now();
	std::cout << "Backprojection took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " ms" << std::endl;

	// check for errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(1);
	}

	cudaMemcpy(voxel_volume, voxel_volume_gpu, nvoxels * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(transient_chunk_gpu);
	cudaFree(T_gpu);
	cudaFree(num_pairs_gpu);
	cudaFree(scanned_pairs_gpu);
	cudaFree(camera_pos_gpu);
	cudaFree(laser_pos_gpu);
	cudaFree(voxel_volume_gpu);
	cudaFree(volume_zero_pos_gpu);
	cudaFree(voxel_inc_gpu);
	cudaFree(t0_gpu);
	cudaFree(deltaT_gpu);
	cudaFree(voxels_per_side_gpu);
	cudaFree(kernel_voxels_gpu);
}