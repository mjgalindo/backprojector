
#include <chrono>
#include <math.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <memory>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <cuComplex.h>
#include "shared_mem.h"

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
    return norm3df(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]);
}

__forceinline__ __device__
void compute_distance(const float * voxel_position,
					  const ppd * pair,
					  float * distance_out)
{
	// From the laser to the wall
	float laser_wall_distance = pair->laser_wall;
	// From the wall to the current voxel
	float laser_point_voxel_distance = distance(pair->laser_point, voxel_position);
	// From the wall back to the camera
	float cam_wall_distance = pair->camera_wall;
	// From the object back to the wall
	float voxel_cam_point_distance = distance(voxel_position, pair->camera_point);
	
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

template <typename FT>
__device__
FT zero_value()
{
	FT zero = 0.0;
	return zero;
}

template <>
__device__
cuComplex zero_value<cuComplex>()
{
	cuComplex zero = make_cuComplex(0.0f, 0.0f);
	return zero;
}

template <typename FT>
__forceinline__ __device__
FT cu_add(FT a, FT b)
{
	return a + b;
}

template <>
__forceinline__ __device__
cuComplex cu_add<cuComplex>(cuComplex a, cuComplex b)
{
	return cuCaddf(a, b);
}

template <typename FT>
__global__
void cuda_backprojection_impl(FT *transient_data,
                              uint32_t *T,
                              uint32_t *num_pairs,
                              ppd *scanned_pairs,
                              FT *voxel_volume,
                              float *volume_zero_pos,
                              float *voxel_inc,
                              float *t0,
							  float *deltaT,
							  float *voxel_footprint,
							  uint32_t *voxels_per_side,
							  uint32_t *kernel_voxels)
{
	uint32_t block_id, voxel_id, xyz[3];
	advance_block(voxels_per_side, kernel_voxels, block_id, xyz, voxel_id);

	// If the block is outside the volume don't do anything
	if ((xyz[0] >= voxels_per_side[0]) | (xyz[1] >= voxels_per_side[1]) | (xyz[2] >= voxels_per_side[2]))
		return;
	
	SharedMemory<FT> local_array_holder;
	FT * local_array = local_array_holder.getPointer();
	FT& radiance_sum = local_array[threadIdx.x];
	radiance_sum = zero_value<FT>();
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
			compute_distance(voxel_position, &scanned_pairs[pair_index], &total_distance);
			int32_t min_time_index = max(   0, (int32_t) round((total_distance - *t0 - *voxel_footprint / 2) / *deltaT));
			int32_t max_time_index = min(*T-1, (int32_t) round((total_distance - *t0 + *voxel_footprint / 2) / *deltaT)) + 1;

			for (int32_t sample_index = min_time_index; sample_index < max_time_index; sample_index++)
			{
				radiance_sum = cu_add(radiance_sum, transient_data[pair_index * *T + sample_index]);
			}
		}
	}
    __syncthreads();
	if (threadIdx.x == 0)
	{
		// Compute the reduction in a single thread and write it
		for (int i = 1; i < blockDim.x; i++) 
		{
			local_array[0] = cu_add(local_array[0], local_array[i]);
		}
		voxel_volume[voxel_id] = local_array[0];
	}
}

__device__
inline bool is_first_thread()
{
	return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0;
}

__global__
void cuda_octree_backprojection_impl(float const *transient_data,
									 uint32_t const *T,
									 uint32_t const *num_pairs,
									 ppd const *scanned_pairs,
									 float *voxel_volume,
									 float const *volume_zero_pos,
									 float const *voxel_inc,
									 float const *t0,
									 float const *deltaT,
									 uint32_t const *voxels_per_side,
									 uint32_t *kernel_voxels,
									 uint32_t const *depth,
									 float const *threshold,
									 float *aux_voxel_volume,
									 float const *parent_aux_voxel_volume,
									 uint32_t *Pp)
{
	bool P = *Pp == 1;

	// TODO: voxels_per_side is not always going to be a cube!!!
	uint32_t mod = voxels_per_side[0] / (*depth);

	uint32_t block_id = (blockIdx.x * gridDim.y * gridDim.z +
						 blockIdx.y * gridDim.z +
						 blockIdx.z) * 3;

	uint32_t xyz[3] = {kernel_voxels[block_id+0], kernel_voxels[block_id+1], kernel_voxels[block_id+2]};

	// Set the next voxel to be computed by this blockIdx in the next call.
	// We advance on the Z axis by the dimensions of the kernel, overflowing to the Y axis and 
	// then to the X axis. This would match row-major access to the 3D array.
	
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
		if (false && P && threadIdx.x == 0)
		printf("XYZ %d, %d, %d  NEXT %d, %d, %d BLCK %d, %d, %d VPS %d, %d, %d\n", xyz[0], xyz[1], xyz[2], next_xyz[0], next_xyz[1], next_xyz[2], blockIdx.x, blockIdx.y, blockIdx.z, voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]);
	}
	__syncthreads();
	
	// If the block is outside the volume don't do anything
	if ((xyz[0] >= voxels_per_side[0]) | (xyz[1] >= voxels_per_side[1]) | (xyz[2] >= voxels_per_side[2]))
	{
		if (threadIdx.x == 0)
			printf("EXITED XYZ %d, %d, %d  BLCK %d, %d, %d VPS %d, %d, %d\n",
				xyz[0], xyz[1], xyz[2], blockIdx.x, blockIdx.y, blockIdx.z, voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]);
		return;
	}
	
	// Return if the current voxel is not considered for the current depth
	if (xyz[0] % mod != 0 || 
		xyz[1] % mod != 0 || 
		xyz[2] % mod != 0)
	{
		if (false && P && threadIdx.x == 0)
		printf("EXITED 2 XYZ %d, %d, %d  BLCK %d, %d, %d VPS %d, %d, %d\n",
			xyz[0], xyz[1], xyz[2], blockIdx.x, blockIdx.y, blockIdx.z, voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]);

		return;
	}

	uint32_t voxel_id = xyz[0] * voxels_per_side[1] * voxels_per_side[2] +
						xyz[1] * voxels_per_side[2] + 
						xyz[2];

	uint32_t aux_xyz[3] = {xyz[0] / mod, xyz[1] / mod, xyz[2] / mod};
	uint32_t aux_voxel_id = aux_xyz[0] * (voxels_per_side[1] / mod) * (voxels_per_side[2] / mod) +
							aux_xyz[1] * (voxels_per_side[2] / mod) + 
							aux_xyz[2];

	uint32_t parent_xyz[3] = {xyz[0] / (mod*2), xyz[1] / (mod*2), xyz[2] / (mod*2)};
	uint32_t parent_voxel_id = parent_xyz[0] * (voxels_per_side[1] / (2*mod)) * (voxels_per_side[2] / (2*mod)) +
						       parent_xyz[1] * (voxels_per_side[2] / (2*mod)) + 
							   parent_xyz[2];
							   
	if (P && threadIdx.x == 0)
	printf("MOD IS %d VPS is %d DEPTH is %d XYZ_ID %d AUX_ID %d PAR_ID %d AUX ADDRESS %p PARENT ADDRESS %p \n",
		mod, voxels_per_side[0], *depth, voxel_id, aux_voxel_id, parent_voxel_id, aux_voxel_volume, parent_aux_voxel_volume);

	if (*depth > 1)
	{
		if (P && threadIdx.x == 0)
		printf("VOX %d, %d, %d PARENT %d %d %d \n", xyz[0], xyz[1], xyz[2], parent_xyz[0], parent_xyz[1], parent_xyz[2]);
		if (parent_aux_voxel_volume[parent_voxel_id] < *threshold)
		{
			aux_voxel_volume[aux_voxel_id] = 0.0f;
			if (P && threadIdx.x == 0)
			printf("EXITED EARLY FROM VOX %d, %d, %d PARENT %d %d %d  PARENT VALUE %.5f < %.5f\n",
				xyz[0], xyz[1], xyz[2], parent_xyz[0], parent_xyz[1], parent_xyz[2], 
				parent_aux_voxel_volume[parent_voxel_id], *threshold);
			return;
		}
	}

	extern __shared__ double local_array[];
	double& radiance_sum = local_array[threadIdx.x];
	radiance_sum = 0.0;

	// Compute size of the voxel for the current depth
	float sizes[3] = {0.0f, 0.0f, 0.0f};
	float voxel_diagonal = 0.0f;
	if (mod != 1)
	{
		for (int i = 0; i < 3; i++) sizes[i] = (voxel_inc[i] * voxels_per_side[i]) / *depth;
		voxel_diagonal = sqrt(sizes[0] * sizes[0] + sizes[1] * sizes[1] + sizes[2] * sizes[2]);
	}
	
	if (P && is_first_thread())
		printf("MAX DIAG %.5f SIZE %.5f voxel_inc %.5f, %.5f, %.5f  VPS %d, %d, %d\n", 
			voxel_diagonal, sizes[0], voxel_inc[0], voxel_inc[1], voxel_inc[2], 
			voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]);

	{
		float origvoxel_position[3] = {
			volume_zero_pos[0]+voxel_inc[0]*xyz[0],
			volume_zero_pos[1]+voxel_inc[1]*xyz[1],
			volume_zero_pos[2]+voxel_inc[2]*xyz[2]};
		float voxel_position[3] = {
			volume_zero_pos[0]+voxel_inc[0]*xyz[0] + sizes[0] / 2,
			volume_zero_pos[1]+voxel_inc[1]*xyz[1] + sizes[1] / 2,
			volume_zero_pos[2]+voxel_inc[2]*xyz[2] + sizes[2] / 2};
		if (P && threadIdx.x == 0)
		{
			printf("XYZ %d %d %d POS %.3f, %.3f, %.3f ORIGPOS %.3f, %.3f, %.3f \n", 
				xyz[0], xyz[1], xyz[2], voxel_position[0], voxel_position[1], voxel_position[2], 
				origvoxel_position[0], origvoxel_position[1], origvoxel_position[2]);
		}
		
		for (uint32_t i = 0; i < *num_pairs / blockDim.x; i++)
		{
			uint32_t pair_index = i * blockDim.x + threadIdx.x;
			float total_distance = 0.0f;
			compute_distance(voxel_position, &scanned_pairs[pair_index], &total_distance);
			int32_t min_time_index = max(0, (int32_t) round((total_distance - *t0 - voxel_diagonal / 2) / *deltaT));
			int32_t max_time_index = min(*T, (int32_t) round((total_distance - *t0 + voxel_diagonal / 2) / *deltaT)) + 1;
			if (P && threadIdx.x == 0 && i == 0)
				printf("Min time: %d Max time: %d total_dist %.3f XYZ %d %d %d POS %.4f, %.4f, %.4f t0 %.4f voxel_diag %.9f voxinc %.5f T %d\n",
					min_time_index, max_time_index, total_distance, xyz[0], xyz[1], xyz[2], *t0, voxel_diagonal, voxel_inc[i], *T);
			if (threadIdx.x == 0 && min_time_index >= max_time_index)
				printf("XYZ %d %d %d POS %.3f, %.3f, %.3f Min time: %d Max time: %d total_dist %.3f t0 %.4f voxel_diag %.9f voxinc %.5f T %d\n",
					xyz[0], xyz[1], xyz[2], voxel_position[0], voxel_position[1], voxel_position[2], 
					min_time_index, max_time_index, total_distance, *t0, voxel_diagonal, voxel_inc[i], *T);
			for (int32_t sample_index = min_time_index; sample_index < max_time_index; sample_index++)
			{
				radiance_sum += transient_data[pair_index * *T + sample_index];
			}
		}
	}
    __syncthreads();
	if (threadIdx.x == 0)
	{
		// Compute the reduction in a single thread and write it
		for (int i = 1; i < blockDim.x; i++) 
		{
			local_array[0] += local_array[i];
		}
		
		if (mod == 1)
		{
			if (P && threadIdx.x == 0)
			printf("FINAL SUM %.7f  XYZ %d %d %d\n", local_array[0], xyz[0], xyz[1], xyz[2]);
			voxel_volume[voxel_id] = (float) local_array[0];
		}
		else
		{
			if (P && threadIdx.x == 0)
			printf("AUX SUM %.7f  XYZ %d %d %d AUX_XYZ %d %d %d PARENT_XYZ %d %d %d\n", 
				local_array[0], xyz[0], xyz[1], xyz[2], aux_xyz[0], aux_xyz[1], aux_xyz[2], parent_xyz[0], parent_xyz[1], parent_xyz[2]);
			aux_voxel_volume[aux_voxel_id] = (float) local_array[0];
		}
	}
    __syncthreads();
}


void call_cuda_backprojection(const float* transient_chunk,
							  uint32_t transient_size, uint32_t T,
							  const std::vector<ppd> scanned_pairs,
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
	thrust::device_vector<ppd> scanned_pairs_gpu(scanned_pairs.begin(), scanned_pairs.end());
	const uint32_t nvoxels = voxels_per_side[0] * voxels_per_side[1] * voxels_per_side[2];
	thrust::device_vector<float> voxel_volume_gpu(voxel_volume, voxel_volume + nvoxels);
	thrust::device_vector<float> volume_zero_pos_gpu(volume_zero_pos, volume_zero_pos + 3);
	thrust::device_vector<float> voxel_inc_gpu(voxel_inc, voxel_inc + 9);
	thrust::device_vector<float> t0_gpu(&t0, &t0 + 1);
	thrust::device_vector<float> deltaT_gpu(&deltaT, &deltaT + 1);
	thrust::device_vector<float> voxel_footprint_gpu(1);
	voxel_footprint_gpu[0] = 0.0f;
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
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_backprojection_impl<float>, sizeof(float), 0); 
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
		cuda_backprojection_impl<<<xyz_blocks, threads_in_block, blockSize*sizeof(float)>>>(
		thrust::raw_pointer_cast(&transient_chunk_gpu[0]),
		thrust::raw_pointer_cast(&T_gpu[0]),
		thrust::raw_pointer_cast(&num_pairs_gpu[0]),
		thrust::raw_pointer_cast(&scanned_pairs_gpu[0]),
		thrust::raw_pointer_cast(&voxel_volume_gpu[0]),
		thrust::raw_pointer_cast(&volume_zero_pos_gpu[0]),
		thrust::raw_pointer_cast(&voxel_inc_gpu[0]),
		thrust::raw_pointer_cast(&t0_gpu[0]),
		thrust::raw_pointer_cast(&deltaT_gpu[0]),
		thrust::raw_pointer_cast(&voxel_footprint_gpu[0]),
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


void call_cuda_octree_backprojection(const float* transient_chunk,
									 uint32_t transient_size, uint32_t T,
									 const std::vector<ppd> scanned_pairs,
									 float* voxel_volume,
									 const uint32_t* voxels_per_side,
									 const float* volume_zero_pos,
									 const float* voxel_inc,
									 float t0,
									 float deltaT)
{
	std::cout << "GPU OCTREE BP NOT WORKING YET\n";
	thrust::device_vector<float> transient_chunk_gpu(transient_chunk, transient_chunk + transient_size);
	thrust::device_vector<uint32_t> T_gpu(&T, &T + 1);
	uint32_t num_pairs = scanned_pairs.size();
	thrust::device_vector<uint32_t> num_pairs_gpu(&num_pairs, &num_pairs + 1);
	thrust::device_vector<ppd> scanned_pairs_gpu(scanned_pairs.begin(), scanned_pairs.end());
	const uint32_t nvoxels = voxels_per_side[0] * voxels_per_side[1] * voxels_per_side[2];
	thrust::device_vector<float> voxel_volume_gpu(voxel_volume, voxel_volume + nvoxels);
	const uint32_t nauxvoxels = voxels_per_side[0] * voxels_per_side[1] * voxels_per_side[2] / 8;
	thrust::device_vector<float> aux_voxel_volume_gpu[2] = {thrust::device_vector<float>(nauxvoxels), thrust::device_vector<float>(nauxvoxels)};
	thrust::fill(aux_voxel_volume_gpu[0].begin(), aux_voxel_volume_gpu[0].end(), 0.0f);
	thrust::fill(aux_voxel_volume_gpu[1].begin(), aux_voxel_volume_gpu[1].end(), 0.0f);
	thrust::device_vector<float> volume_zero_pos_gpu(volume_zero_pos, volume_zero_pos + 3);
	thrust::device_vector<float> voxel_inc_gpu(voxel_inc, voxel_inc + 3);
	thrust::device_vector<float> t0_gpu(&t0, &t0 + 1);
	thrust::device_vector<float> deltaT_gpu(&deltaT, &deltaT + 1);
	thrust::device_vector<uint32_t> voxels_per_side_gpu(voxels_per_side, voxels_per_side + 3);

	float threshold = std::stof(std::getenv("T")); // TODO: DONT FORCE EVERYTHING TO BE COMPUTED
	thrust::device_vector<float> threshold_gpu(&threshold, &threshold+1);

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
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_backprojection_impl<float>, sizeof(float), 0); 
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

	dim3 xyz_blocks(minGridSize, minGridSize, minGridSize);
	dim3 threads_in_block(blockSize, 1, 1);
	uint32_t number_of_runs = std::ceil(std::max(std::max(voxels_per_side[0], voxels_per_side[1]), voxels_per_side[2]) / (float) minGridSize);
	number_of_runs = number_of_runs * number_of_runs * number_of_runs;

	std::cout << "Backprojecting on the GPU using the \"optimal\" configuration" << std::endl;
	std::cout << "# Blocks: " << xyz_blocks.x << ' ' << xyz_blocks.y << ' ' << xyz_blocks.z << std::endl;
	std::cout << "# Threads per block: " << threads_in_block.x << ' ' << threads_in_block.y << ' ' << threads_in_block.z << std::endl;
	std::cout << "# Kernel calls: " << number_of_runs << std::endl;
	
	auto start = std::chrono::steady_clock::now();
	if (voxels_per_side[0] != voxels_per_side[1] || 
		voxels_per_side[1] != voxels_per_side[2] || 
		voxels_per_side[0] != voxels_per_side[2]){
		std::cerr << "USE A CUBE, AND MAKE IT A POWER OF 2!!\n";
		exit(1);
	}

	uint32_t *depth_gpu;
	cudaMalloc((void **)&depth_gpu, 1*sizeof(uint32_t));

	//////
	float* aux_vol_tmp = new float[nauxvoxels];
	//////
	uint32_t Pval = std::stoi(std::getenv("P"));
	std::cout << "P IS " << Pval << std::endl;
	uint32_t *Pp_gpu;
	cudaMalloc((void **)&Pp_gpu, sizeof(uint32_t));
	cudaMemcpy(Pp_gpu, &Pval, sizeof(uint32_t), cudaMemcpyHostToDevice);
	for (uint32_t depth = 1; depth <= voxels_per_side[0]; depth*=2)
	{
		std::cout << "DEPTH " << depth << " T " << T << " dT " << deltaT << std::endl;
		cudaMemcpy(depth_gpu, &depth, sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(kernel_voxels_gpu, kernel_voxels.data(), 3*num_blocks_per_kernel_run*sizeof(uint32_t), cudaMemcpyHostToDevice);
		#if __linux__
		tqdm bar;
		bar.set_theme_braille();
		#else
		std::cout << 0 << " / " << number_of_runs << std::flush;
		#endif
		for (uint32_t r = 0; r < number_of_runs; r++)
		{
			auto start = std::chrono::steady_clock::now();
			cuda_octree_backprojection_impl<<<xyz_blocks, threads_in_block, blockSize*sizeof(double)>>>(
				thrust::raw_pointer_cast(&transient_chunk_gpu[0]),
				thrust::raw_pointer_cast(&T_gpu[0]),
				thrust::raw_pointer_cast(&num_pairs_gpu[0]),
				thrust::raw_pointer_cast(&scanned_pairs_gpu[0]),
				thrust::raw_pointer_cast(&voxel_volume_gpu[0]),
				thrust::raw_pointer_cast(&volume_zero_pos_gpu[0]),
				thrust::raw_pointer_cast(&voxel_inc_gpu[0]),
				thrust::raw_pointer_cast(&t0_gpu[0]),
				thrust::raw_pointer_cast(&deltaT_gpu[0]),
				thrust::raw_pointer_cast(&voxels_per_side_gpu[0]),
				thrust::raw_pointer_cast(&kernel_voxels_gpu[0]),
				depth_gpu,
				thrust::raw_pointer_cast(&threshold_gpu[0]),
				thrust::raw_pointer_cast(&aux_voxel_volume_gpu[0][0]),
				thrust::raw_pointer_cast(&aux_voxel_volume_gpu[1][0]),
				Pp_gpu);
			cudaDeviceSynchronize();
			#if __linux__
			bar.progress(r, number_of_runs);
			#else
			std::cout << '\r' << r+1 << " / " << number_of_runs << std::flush;
			#endif
		}
		std::swap(aux_voxel_volume_gpu[0], aux_voxel_volume_gpu[1]);		
		#if __linux__
		bar.finish();
		#else
		std::cout << std::endl;
		#endif
		std::ofstream wf("depth_" + std::to_string(depth) + ".vol", std::ios::out | std::ios::binary);
		thrust::copy(aux_voxel_volume_gpu[1].begin(), aux_voxel_volume_gpu[1].end(), aux_vol_tmp);
		wf.write((char*) aux_vol_tmp, nauxvoxels*sizeof(float));
	}
	
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

void call_cuda_complex_backprojection(const std::complex<float>* transient_chunk,
											uint32_t transient_size, uint32_t T,
											const std::vector<ppd> scanned_pairs,
											std::complex<float>* voxel_volume,
											const uint32_t* voxels_per_side,
											const float* volume_zero_pos,
											const float* voxel_inc,
											float t0,
											float deltaT)
{
	std::cout << transient_size << ' ' << T << ' ' << scanned_pairs.size() << ' ' << t0 << ' ' << deltaT << std::endl;
	thrust::device_vector<cuComplex> transient_chunk_gpu((cuComplex*) transient_chunk, (cuComplex*) transient_chunk + transient_size);
	thrust::device_vector<uint32_t> T_gpu(&T, &T + 1);
	uint32_t num_pairs = scanned_pairs.size();
	thrust::device_vector<uint32_t> num_pairs_gpu(&num_pairs, &num_pairs + 1);
	thrust::device_vector<ppd> scanned_pairs_gpu(scanned_pairs.begin(), scanned_pairs.end());
	const uint32_t nvoxels = voxels_per_side[0] * voxels_per_side[1] * voxels_per_side[2];
	thrust::device_vector<cuComplex> voxel_volume_gpu((cuComplex*)voxel_volume, ((cuComplex*)voxel_volume) + nvoxels);
	thrust::device_vector<float> volume_zero_pos_gpu(volume_zero_pos, volume_zero_pos + 3);
	thrust::device_vector<float> voxel_inc_gpu(voxel_inc, voxel_inc + 9);
	thrust::device_vector<float> t0_gpu(&t0, &t0 + 1);
	thrust::device_vector<float> deltaT_gpu(&deltaT, &deltaT + 1);
	thrust::device_vector<float> voxel_footprint_gpu(1);
	voxel_footprint_gpu[0] = 0.0f;
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
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_backprojection_impl<cuComplex>, sizeof(cuComplex), 0); 
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
		cuda_backprojection_impl<<<xyz_blocks, threads_in_block, blockSize*sizeof(cuComplex)>>>(
			thrust::raw_pointer_cast(&transient_chunk_gpu[0]),
			thrust::raw_pointer_cast(&T_gpu[0]),
			thrust::raw_pointer_cast(&num_pairs_gpu[0]),
			thrust::raw_pointer_cast(&scanned_pairs_gpu[0]),
			thrust::raw_pointer_cast(&voxel_volume_gpu[0]),
			thrust::raw_pointer_cast(&volume_zero_pos_gpu[0]),
			thrust::raw_pointer_cast(&voxel_inc_gpu[0]),
			thrust::raw_pointer_cast(&t0_gpu[0]),
			thrust::raw_pointer_cast(&deltaT_gpu[0]),
			thrust::raw_pointer_cast(&voxel_footprint_gpu[0]),
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
	{
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(1);
		}
	}
	thrust::copy(voxel_volume_gpu.begin(), voxel_volume_gpu.end(), (cuComplex*)voxel_volume);
}
