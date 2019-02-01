
#include <chrono>
#include <math.h>
#include <iostream>
#include <chrono>

#include <helper_cuda.h>
#include "backproject_cuda.hpp"

__device__
float distance(const float *p1, const float *p2) 
{
    float buff[3]; 
    for (int i = 0; i < 3; i++)
    {
        buff[i] = p1[i] - p2[i];
    }

    return norm3df(buff[0], buff[1], buff[2]);
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
	// First needs to find the x y z coordinates of the voxel
	uint32_t block_id = (blockIdx.x * MAX_BLOCKS_PER_KERNEL_RUN * MAX_BLOCKS_PER_KERNEL_RUN + 
						 blockIdx.y * MAX_BLOCKS_PER_KERNEL_RUN +
						 blockIdx.z) * 3;
	uint32_t xyz[3] = {kernel_voxels[block_id], kernel_voxels[block_id+1], kernel_voxels[block_id+2]};

	{
		// Set the next voxel to be computed by this blockIdx in the next call
		uint32_t* next_xyz = &kernel_voxels[block_id];
		next_xyz[2] = xyz[2] + gridDim.z;
		if (next_xyz[2] >= voxels_per_side[2])
		{
			next_xyz[2] = next_xyz[2] % voxels_per_side[0];
			next_xyz[1] = next_xyz[1] + gridDim.y;
			if (next_xyz[1] >= voxels_per_side[1])
			{
				next_xyz[1] = next_xyz[1] % voxels_per_side[1];
				next_xyz[0] = next_xyz[0] + gridDim.x;
			}
		}
		//if (threadIdx.x == 0) printf("%d %d %d -> %d %d %d\n", xyz[0], xyz[1], xyz[2], next_xyz[0], next_xyz[1], next_xyz[2]);
	}

	uint32_t voxel_id = xyz[0] * voxels_per_side[1] * voxels_per_side[2] + xyz[1] * voxels_per_side[2] + xyz[2];
	// Don't run if the current voxel is not 0. This means the current block has already finished.
	if (voxel_volume[voxel_id] > 0.0)
	{
		return;
	}

    __shared__ double local_array[MAX_THREADS_PER_BLOCK];
    double& radiance_sum = local_array[threadIdx.x];
    radiance_sum = 0.0;

    float voxel_position[] = {volume_zero_pos[0]+voxel_inc[0]*xyz[0],
                              volume_zero_pos[1]+voxel_inc[1]*xyz[1],
                              volume_zero_pos[2]+voxel_inc[2]*xyz[2]};

    for (uint32_t i = 0; i < *num_pairs / MAX_THREADS_PER_BLOCK; i++)
    {
        uint32_t pair_index = i * MAX_THREADS_PER_BLOCK + threadIdx.x;
           
        const pointpair& pair = scanned_pairs[pair_index];

        // From the laser to the wall
        float laser_wall_distance = distance(laser_pos, pair.laser_point);
        
        // From the wall to the current voxel
        float laser_point_voxel_distance = distance(pair.laser_point, voxel_position);
        
        // From the wall back to the camera
        float cam_wall_distance = distance(pair.cam_point, camera_pos);
        // From the object back to the wall
        float voxel_cam_point_distance = distance(voxel_position, pair.cam_point);

        // Radiance gets attenuated with the square of traveled distance between bounces
        /*float distance_attenuation = laser_wall_distance * laser_wall_distance +
                                     laser_point_voxel_distance * laser_point_voxel_distance +
                                     voxel_cam_point_distance * voxel_cam_point_distance +
                                     cam_wall_distance * cam_wall_distance;*/
        // TODO: Cosine attenuation due to Lambert's law
        
        float total_distance = laser_wall_distance + laser_point_voxel_distance + voxel_cam_point_distance + cam_wall_distance;
        
        uint32_t time_index = round((total_distance - *t0) / *deltaT);
        uint32_t tdindex = pair_index * *T + time_index;

        radiance_sum += transient_data[tdindex]; // * distance_attenuation;
    }

    __syncthreads();
    double total_radiance = 0.0;
    for (int i = 0; i < MAX_THREADS_PER_BLOCK; i++) {
        total_radiance += local_array[i];
    }

    // All threads write the same value
	voxel_volume[voxel_id] = total_radiance;
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
                              uint32_t t0,
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
	float *camera_pos_gpu;
	cudaMalloc((void **)&camera_pos_gpu, 3 * sizeof(float));
	cudaMemcpy(camera_pos_gpu, camera_position, 3 * sizeof(float), cudaMemcpyHostToDevice);
	/// float *laser_pos,
	float *laser_pos_gpu;
	cudaMalloc((void **)&laser_pos_gpu, 3 * sizeof(float));
	cudaMemcpy(laser_pos_gpu, laser_position, 3 * sizeof(float), cudaMemcpyHostToDevice);
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
	cudaMalloc((void **)&t0_gpu, sizeof(uint32_t));
	cudaMemcpy(t0_gpu, &T, sizeof(uint32_t), cudaMemcpyHostToDevice);
	/// float *deltaT,
	float *deltaT_gpu;
	cudaMalloc((void **)&deltaT_gpu, sizeof(float));
	cudaMemcpy(deltaT_gpu, &deltaT, sizeof(float), cudaMemcpyHostToDevice);
	/// uint32_t *voxels_per_side,
	uint32_t *voxels_per_side_gpu;
	cudaMalloc((void **)&voxels_per_side_gpu, 3 * sizeof(uint32_t));
    cudaMemcpy(voxels_per_side_gpu, voxels_per_side, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	std::vector<uint32_t> kernel_voxels(MAX_BLOCKS_PER_KERNEL_RUN * MAX_BLOCKS_PER_KERNEL_RUN * MAX_BLOCKS_PER_KERNEL_RUN * 3);
	for (int x = 0; x < MAX_BLOCKS_PER_KERNEL_RUN; x++)
	for (int y = 0; y < MAX_BLOCKS_PER_KERNEL_RUN; y++)
	for (int z = 0; z < MAX_BLOCKS_PER_KERNEL_RUN; z++)
	{
		kernel_voxels[(x*MAX_BLOCKS_PER_KERNEL_RUN*MAX_BLOCKS_PER_KERNEL_RUN+y*MAX_BLOCKS_PER_KERNEL_RUN+z)*3 + 0] = x;
		kernel_voxels[(x*MAX_BLOCKS_PER_KERNEL_RUN*MAX_BLOCKS_PER_KERNEL_RUN+y*MAX_BLOCKS_PER_KERNEL_RUN+z)*3 + 1] = y;
		kernel_voxels[(x*MAX_BLOCKS_PER_KERNEL_RUN*MAX_BLOCKS_PER_KERNEL_RUN+y*MAX_BLOCKS_PER_KERNEL_RUN+z)*3 + 2] = z;
	}

	uint32_t *kernel_voxels_gpu;
	const uint32_t num_blocks_per_kernel_run = MAX_BLOCKS_PER_KERNEL_RUN*MAX_BLOCKS_PER_KERNEL_RUN*MAX_BLOCKS_PER_KERNEL_RUN;
	cudaMalloc((void **)&kernel_voxels_gpu, 3*num_blocks_per_kernel_run*sizeof(uint32_t));
	cudaMemcpy(kernel_voxels_gpu, kernel_voxels.data(), 3*num_blocks_per_kernel_run*sizeof(uint32_t), cudaMemcpyHostToDevice);
	// Ideally, we would run {voxels_per_side[0], voxels_per_side[1], voxels_per_side[2]} 
	// blocks, but windows doesn't like this, so we run many smaller kernels.

	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the 
					// maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 

	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, cuda_backprojection_impl, 0, 0); 
	std::cout << "JUST SAYIN: " << minGridSize << ' ' << blockSize << std::endl;

	dim3 dimBlock(MAX_BLOCKS_PER_KERNEL_RUN, MAX_BLOCKS_PER_KERNEL_RUN, MAX_BLOCKS_PER_KERNEL_RUN);
	dim3 dimThreads(MAX_THREADS_PER_BLOCK, 1, 1);

	std::cout << "Backprojecting on the GPU" << std::endl;
	std::cout << "BLOCKS: " << dimBlock.x << ' ' << dimBlock.y << ' ' << dimBlock.z << std::endl;
	std::cout << "THREADS: " << dimThreads.x << ' ' << dimThreads.y << ' ' << dimThreads.z << std::endl;

	uint32_t number_of_runs = std::ceil(voxels_per_side[0] / (float) MAX_BLOCKS_PER_KERNEL_RUN);
	number_of_runs = number_of_runs * number_of_runs * number_of_runs;
	for (uint32_t r = 0; r < number_of_runs; r++)
	{
		auto start = std::chrono::steady_clock::now();
		cuda_backprojection_impl<<<dimBlock, dimThreads>>>(transient_chunk_gpu,
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
		auto end = std::chrono::steady_clock::now();
		std::cout << "Elapsed time in milliseconds : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
	}

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