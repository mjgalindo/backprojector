
#include <chrono>
#include <math.h>
#include <iostream>
#include <fstream>

#include <thread>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <cuComplex.h>

#ifdef _WIN32
#define MAKE_DLL_EXPORT __declspec(dllexport)
#endif
#include "backproject_cuda.hpp"
#if __linux__
#include "tqdm.h"
#endif

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

	{
		float voxel_position[] = {
			volume_zero_pos[0] + voxel_inc[0] * xyz[0] + voxel_inc[3] * xyz[1] + voxel_inc[6] * xyz[2],
			volume_zero_pos[1] + voxel_inc[1] * xyz[0] + voxel_inc[4] * xyz[1] + voxel_inc[7] * xyz[2],
			volume_zero_pos[2] + voxel_inc[2] * xyz[0] + voxel_inc[5] * xyz[1] + voxel_inc[8] * xyz[2]
		};

		for (uint32_t i = 0; i < *num_pairs / blockDim.x; i++)
		{
			uint32_t pair_index = i * blockDim.x + threadIdx.x;
			float total_distance;
			compute_distance(voxel_position, laser_pos, camera_pos, &scanned_pairs[pair_index], &total_distance);
			uint32_t time_index = round((total_distance - *t0) / *deltaT);
			if (time_index < *T)
			{
				uint32_t tdindex = pair_index * *T + time_index;
				radiance_sum += transient_data[tdindex]; // * distance_attenuation;
			}
		}
	}
    __syncthreads();
	if (threadIdx.x == 0)
	{
		// Compute the reduction in a single thread and write it
		for (int i = 1; i < blockDim.x; i++) {
			local_array[0] += local_array[i];
		}
		voxel_volume[voxel_id] = (float) local_array[0];
	}
    __syncthreads();
}

__global__
void cuda_complex_backprojection_impl(cuComplex *transient_data,
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
		
	extern __shared__ cuDoubleComplex local_array2[];
	cuDoubleComplex& radiance_sum = local_array2[threadIdx.x];
	radiance_sum = make_cuDoubleComplex(0.0, 0.0);

	
	// Don't run if the current voxel is not 0. This means the current block has already finished.
	if (voxel_volume[voxel_id] == 0.0)
	{
		float voxel_position[3] = {
			volume_zero_pos[0] + voxel_inc[0] * xyz[0] + voxel_inc[3] * xyz[1] + voxel_inc[6] * xyz[2],
			volume_zero_pos[1] + voxel_inc[1] * xyz[0] + voxel_inc[4] * xyz[1] + voxel_inc[7] * xyz[2],
			volume_zero_pos[2] + voxel_inc[2] * xyz[0] + voxel_inc[5] * xyz[1] + voxel_inc[8] * xyz[2]};

		for (uint32_t i = 0; i < *num_pairs / blockDim.x; i++)
		{
			uint32_t pair_index = i * blockDim.x + threadIdx.x;
			float total_distance;
			compute_distance(voxel_position, laser_pos, camera_pos, &scanned_pairs[pair_index], &total_distance);
			uint32_t time_index = round((total_distance - *t0) / *deltaT);
			uint32_t tdindex = pair_index * *T + time_index;

			radiance_sum = cuCadd(radiance_sum, cuComplexFloatToDouble(transient_data[tdindex])); // * distance_attenuation;
		}
	}
    __syncthreads();
	if (threadIdx.x == 0)
	{
		// Compute the reduction in a single thread and write it
		for (int i = 1; i < blockDim.x; i++) {
			local_array2[0] = cuCadd(local_array2[0], local_array2[i]);
		}
		voxel_volume[voxel_id] = cuCreal(local_array2[0]);
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
	thrust::device_vector<float> transient_chunk_gpu(transient_chunk, transient_chunk + transient_size);
	thrust::device_vector<uint32_t> T_gpu(&T, &T + 1);
	uint32_t num_pairs = scanned_pairs.size();
	thrust::device_vector<uint32_t> num_pairs_gpu(&num_pairs, &num_pairs + 1);
	thrust::device_vector<pointpair> scanned_pairs_gpu(scanned_pairs.begin(), scanned_pairs.end());
	thrust::device_vector<float> camera_pos_gpu(camera_position, camera_position + 3);
	thrust::device_vector<float> laser_pos_gpu(laser_position, laser_position + 3);
	const uint32_t nvoxels = voxels_per_side[0] * voxels_per_side[1] * voxels_per_side[2];
	thrust::device_vector<float> voxel_volume_gpu(voxel_volume, voxel_volume + nvoxels);
	thrust::device_vector<float> volume_zero_pos_gpu(volume_zero_pos, volume_zero_pos + 3);
	thrust::device_vector<float> voxel_inc_gpu(voxel_inc, voxel_inc + 9);
	thrust::device_vector<float> t0_gpu(&t0, &t0 + 1);
	thrust::device_vector<float> deltaT_gpu(&deltaT, &deltaT + 1);
	thrust::device_vector<uint32_t> voxels_per_side_gpu(voxels_per_side, voxels_per_side + 3);

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
	#if __linux__
	tqdm bar;
	bar.set_theme_braille();
	#else
	std::cout << 0 << " / " << number_of_runs << std::flush;
	#endif
	for (uint32_t r = 0; r < number_of_runs; r++)
	{
		auto start = std::chrono::steady_clock::now();
		cuda_backprojection_impl<<<xyz_blocks, threads_in_block, blockSize*sizeof(double)>>>(
			thrust::raw_pointer_cast(&transient_chunk_gpu[0]),
			thrust::raw_pointer_cast(&T_gpu[0]),
			thrust::raw_pointer_cast(&num_pairs_gpu[0]),
			thrust::raw_pointer_cast(&scanned_pairs_gpu[0]),
			thrust::raw_pointer_cast(&camera_pos_gpu[0]),
			thrust::raw_pointer_cast(&laser_pos_gpu[0]),
			thrust::raw_pointer_cast(&voxel_volume_gpu[0]),
			thrust::raw_pointer_cast(&volume_zero_pos_gpu[0]),
			thrust::raw_pointer_cast(&voxel_inc_gpu[0]),
			thrust::raw_pointer_cast(&t0_gpu[0]),
			thrust::raw_pointer_cast(&deltaT_gpu[0]),
			thrust::raw_pointer_cast(&voxels_per_side_gpu[0]),
			thrust::raw_pointer_cast(&kernel_voxels_gpu[0]));

		cudaDeviceSynchronize();
		#if __linux__
		bar.progress(r, number_of_runs);
		#else
		std::cout << '\r' << r+1 << " / " << number_of_runs << std::flush;
		#endif
	}
	cudaDeviceSynchronize();
	#if __linux__
	bar.finish();
	#else
	std::cout << std::endl;
	#endif
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
	
	thrust::copy(voxel_volume_gpu.begin(), voxel_volume_gpu.end(), voxel_volume);
}


__global__
void convolve1d(const float* transient_in,
				const uint32_t* transient_shape,
				const float* kernel,
				const uint32_t* kernel_size,
				float* convolved_out)
{
	const int num_points = transient_shape[0];
	const int T = transient_shape[1];

	for (int tId = 0; tId < T / blockDim.x; tId++)
	{
		const int id = blockIdx.x*num_points+blockIdx.x+tId*threadIdx.x;
		convolved_out[id] = 0;
		for (int k = 0; k < *kernel_size; k++)
		{
			convolved_out[id] += transient_in[id] * kernel[k];
		}
	}
}

void call_cuda_convolve1d(const float* transient_in,
						  const uint32_t* transient_shape,
						  const float* kernel,
						  const uint32_t kernel_size,
						  float* convolved_out)
{
	const uint32_t transient_total = transient_shape[0]*transient_shape[1];
	thrust::device_vector<float> transient_in_gpu(transient_in, transient_in + transient_total);
	thrust::device_vector<uint32_t> transient_shape_gpu(transient_shape, transient_shape + 2);
	thrust::device_vector<float> kernel_gpu(kernel, kernel + kernel_size);
	thrust::device_vector<uint32_t> kernel_size_gpu(&kernel_size, &kernel_size + 1);
	thrust::device_vector<float> convolved_out_gpu(transient_total);
	std::cout << transient_shape[0] << ' ' << transient_shape[1] << std::endl;
	std::cout << kernel_size << std::endl;
	dim3 xyz_blocks(transient_shape[0], 1, 1);
	dim3 threads_in_block(transient_shape[1], 1, 1);
	std::cout << "CONVOLUTION DONE " << transient_in << " " << convolved_out << " " << convolved_out[0] << " " << convolved_out_gpu[0] << std::endl;
	convolve1d<<<xyz_blocks, threads_in_block>>>(
		thrust::raw_pointer_cast(&transient_in_gpu[0]),
		thrust::raw_pointer_cast(&transient_shape_gpu[0]),
		thrust::raw_pointer_cast(&kernel_gpu[0]),
		thrust::raw_pointer_cast(&kernel_size_gpu[0]),
		thrust::raw_pointer_cast(&convolved_out_gpu[0]));
	std::cout << "CONVOLUTION DONE " << transient_in << " " << convolved_out << " " << convolved_out[0] << " " << convolved_out_gpu[0] << std::endl;

	thrust::copy(thrust::raw_pointer_cast(&convolved_out_gpu[0]), thrust::raw_pointer_cast(&convolved_out_gpu[0])+transient_total, convolved_out);
	std::cout << "COPY DONE " << convolved_out[0] << std::endl;
}
