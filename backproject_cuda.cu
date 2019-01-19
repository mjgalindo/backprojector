
#include <chrono>
#include <math.h>
#include <helper_cuda.h>

#include "backproject_cuda.hpp"

namespace bp_cuda
{
    const uint MAX_THREADS_PER_BLOCK = 4;

    __device__
    float distance(float *p1, float *p2) 
    {
        float buff[3]; 
        for (int i = 0; i < 3; i++)
        {
            buff[i] = p1[i] - p2[i];
            buff[i] = buff[i] * buff[i];
        }

        return sqrt(buff[0]+buff[1]+buff[2]);
    }

    __global__
    void cuda_backprojection(float *transient_data,
                             uint *T,
                             uint *laser_grid_points,
                             uint *camera_grid_points,
                             float *camera_grid_positions,
                             float *laser_grid_positions,
                             float *camera_pos,
                             float *laser_pos,
                             float *voxel_volume, 
                             float *voxel_positions, 
                             float *t0, 
                             float *deltaT,
                             uint *voxels_per_side,
                             bool *is_confocal)
    {
        uint voxel_id = blockIdx.x * *voxels_per_side * *voxels_per_side + blockIdx.y * *voxels_per_side + blockIdx.z;
        __shared__ float local_array[MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK]; 
        __shared__ uint iterations[MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK]; 
        float& radiance_sum = local_array[threadIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.y];
        uint& its = iterations[threadIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.y];
        // Ensure it's 0 initialized
        radiance_sum = 0.0;
        float *voxel_position = &voxel_positions[voxel_id*3];
        its = 0;
        for (uint cxi = 0; cxi < camera_grid_points[0] / MAX_THREADS_PER_BLOCK; cxi++)
        {
        uint cx = cxi * camera_grid_points[0] / MAX_THREADS_PER_BLOCK + threadIdx.x;
        //printf("CX: %d, CXi: %d, Tx: %d Ty: %d, Its: %d \ncxi %d, cgp: %d, mtpb: %d, tx: %d\n", cx, cxi, threadIdx.x, threadIdx.y, its, cxi, camera_grid_points[0], MAX_THREADS_PER_BLOCK, threadIdx.x);
        //printf("cxi %d, cgp: %d, mtpb: %d, tx: %d\n", cxi, camera_grid_points[0], MAX_THREADS_PER_BLOCK, threadIdx.x);
        for (uint cyi = 0; cyi < camera_grid_points[1] / MAX_THREADS_PER_BLOCK; cyi++)
        {
            uint cy = cyi * camera_grid_points[1] / MAX_THREADS_PER_BLOCK + threadIdx.y;
            //printf("CX: %d, CY: %d CXi: %d, CYi: %d  Tx: %d Ty: %d, Its: %d \n", cx, cy, cxi, cyi, threadIdx.x, threadIdx.y, its);
            float *camera_wall_point = &camera_grid_positions[(cx*camera_grid_points[1] + cy)*3];
            float camera_wall_distance = distance(camera_pos, camera_wall_point);
            float voxel_camera_point_distance = distance(camera_wall_point, voxel_position);
            its++;

            if (!*is_confocal)
            {
                for (uint lx = 0; lx < laser_grid_points[0]; lx++)
                {
                for (uint ly = 0; ly < laser_grid_points[1]; ly++)
                {   
                    float *laser_wall_point = &laser_grid_positions[(lx*laser_grid_points[1] + ly)*3];
                    float laser_wall_distance = distance(laser_pos, laser_wall_point);

                    float wall_voxel_wall_distance = distance(laser_wall_point, voxel_position) + 
                                                     voxel_camera_point_distance;
                    float total_distance = laser_wall_distance + wall_voxel_wall_distance + camera_wall_distance;
                    uint time_index = round((total_distance - *t0) / *deltaT);
                    if (time_index < *T)
                    {
                        // transient_data.shape() -> (T, cy, cx, ly, lx)
                        radiance_sum += transient_data[time_index * camera_grid_points[1] * camera_grid_points[0]
                            * laser_grid_points[1] * laser_grid_points[0] + cy * camera_grid_points[0] * laser_grid_points[1] * laser_grid_points[0] +
                            cx * laser_grid_points[1] * laser_grid_points[0] + ly * laser_grid_points[0] + lx];
                    }
                }
            }
            } else 
            {
                float wall_voxel_wall_distance = voxel_camera_point_distance*2;
                float total_distance = camera_wall_distance + wall_voxel_wall_distance + camera_wall_distance;
                uint time_index = round((total_distance - *t0) / *deltaT);
                if (time_index < *T)
                {
                    radiance_sum += transient_data[time_index * camera_grid_points[1] * camera_grid_points[0] + cy * camera_grid_points[0] + cx];
                }
            }
        }
        }
        __syncthreads();
        float total_radiance = 0.0;
        for (int i = 0; i < MAX_THREADS_PER_BLOCK*MAX_THREADS_PER_BLOCK; i++) {
            //printf("LOCAL ARRAY B{%d %d %d} %d %.7f\n", blockIdx.x, blockIdx.y, blockIdx.z, i, local_array[i]);
            total_radiance += local_array[i];
        }
        //printf("TOTAL B{%d %d %d} %.7f %.7f\n", blockIdx.x, blockIdx.y, blockIdx.z, radiance_sum, local_array[0]);
        // All threads write the same value
        //printf("%x\n", &voxel_volume[voxel_id] );
        voxel_volume[voxel_id] = total_radiance;
        //printf("B{%d %d %d} t{%d %d}: RS: %.7f  LA: %.7f %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, radiance_sum, local_array[threadIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.y], its);

        // __syncthreads();
    }


    xt::xarray<float> backproject(
        const xt::xarray<float>& transient_data,
        const xt::xtensor<float, 3>& dcamera_grid_positions,
        const xt::xtensor<float, 3>& dlaser_grid_positions,
        vector3 camera_position,
        vector3 laser_position,
        float t0,
        float deltaT,
        bool is_confocal,
        vector3 volume_position,
        float volume_size,
        uint voxels_per_side)
    {
        float voxel_size = volume_size / (voxels_per_side - 1);

        xt::xtensor<float, 3> camera_grid_positions = dcamera_grid_positions;
        xt::xtensor<float, 3> laser_grid_positions = dlaser_grid_positions;
        
        // Ensure points are defined so that (x, y, z) are contiguous in memory
        if (camera_grid_positions.shape()[2] != 3) {
            camera_grid_positions = xt::transpose(camera_grid_positions, {2,1,0});
        }
        if (laser_grid_positions.shape()[2] != 3) {
            laser_grid_positions = xt::transpose(laser_grid_positions, {2,1,0});
        }

        auto get_point = [volume_position, volume_size, voxel_size](uint x, uint y, uint z) -> vector3 {
            static auto zero_pos = volume_position - volume_size / 2;
            return zero_pos + vector3{x * voxel_size, y * voxel_size, z * voxel_size};
        };

        std::array<uint, 2> camera_grid_points;
        std::array<uint, 2> laser_grid_points;
        {
            // camera_grid_positions.shape() -> (points_x, points_y, 3)
            auto t = camera_grid_positions.shape();
            camera_grid_points[0] = t[0];
            camera_grid_points[1] = t[1];

            // laser_grid_positions.shape() -> (points_x, points_y, 3)
            t = laser_grid_positions.shape();
            laser_grid_points[0] = t[0];
            laser_grid_points[1] = t[1];
        }

        uint T = transient_data.shape()[0];
        uint total_transient_size = sizeof(float);
        for (const auto d : transient_data.shape()) {
            total_transient_size *= d;
        }

        xt::xarray<float> voxel_volume = xt::zeros<float>({voxels_per_side, voxels_per_side, voxels_per_side});
        xt::xarray<float> voxel_positions = xt::zeros<float>({voxels_per_side, voxels_per_side, voxels_per_side, 3u});

        for (uint x = 0; x < voxels_per_side; x++)
        for (uint y = 0; y < voxels_per_side; y++)
        for (uint z = 0; z < voxels_per_side; z++)
            xt::view(voxel_positions, x,y,z, xt::all()) = get_point(x,y,z);

        // Copy all the necessary information to the device

        /// float *transient_data,
        float *transient_data_gpu;
        cudaMalloc((void**)&transient_data_gpu, total_transient_size);
	    cudaMemcpy(transient_data_gpu, transient_data.data(), total_transient_size, cudaMemcpyHostToDevice); 
        /// uint *T,
        uint *T_gpu;
        cudaMalloc((void**)&T_gpu, sizeof(uint));
        cudaMemcpy(T_gpu, &T, sizeof(uint), cudaMemcpyHostToDevice);
        /// uint *laser_grid_points,
        uint *laser_grid_points_gpu;
        cudaMalloc((void**)&laser_grid_points_gpu, 2*sizeof(uint));
        cudaMemcpy(laser_grid_points_gpu, laser_grid_points.data(), 2*sizeof(uint), cudaMemcpyHostToDevice);
        /// uint *camera_grid_points,
        uint *camera_grid_points_gpu;
        cudaMalloc((void**)&camera_grid_points_gpu, 2*sizeof(uint));
        cudaMemcpy(camera_grid_points_gpu, camera_grid_points.data(), 2*sizeof(uint), cudaMemcpyHostToDevice);
        /// float *camera_grid_positions,
        float *camera_grid_positions_gpu;
        uint cgpos_size = camera_grid_points[0]*camera_grid_points[1]*3*sizeof(float);
        cudaMalloc((void**)&camera_grid_positions_gpu, cgpos_size);
        cudaMemcpy(camera_grid_positions_gpu, camera_grid_positions.data(), cgpos_size, cudaMemcpyHostToDevice);
        /// float *laser_grid_positions,
        float *laser_grid_positions_gpu;
        uint lgpos_size = laser_grid_points[0]*laser_grid_points[1]*3*sizeof(float);
        cudaMalloc((void**)&laser_grid_positions_gpu, lgpos_size);
        cudaMemcpy(laser_grid_positions_gpu, laser_grid_positions.data(), lgpos_size, cudaMemcpyHostToDevice);
        /// float *camera_pos,
        float *camera_pos_gpu;
        cudaMalloc((void**)&camera_pos_gpu, 3*sizeof(float));
        cudaMemcpy(camera_pos_gpu, camera_position.data(), 3*sizeof(float), cudaMemcpyHostToDevice); 
        /// float *laser_pos,
        float *laser_pos_gpu;
        cudaMalloc((void**)&laser_pos_gpu, 3*sizeof(float));
        cudaMemcpy(laser_pos_gpu, laser_position.data(), 3*sizeof(float), cudaMemcpyHostToDevice); 
        /// float *voxel_volume, 
        float *voxel_volume_gpu;
        uint nvoxels = voxels_per_side*voxels_per_side*voxels_per_side;
        cudaMalloc((void**)&voxel_volume_gpu, nvoxels*sizeof(float));
        cudaMemcpy(voxel_volume_gpu, voxel_volume.data(), nvoxels*sizeof(float), cudaMemcpyHostToDevice); 
        /// float *voxel_positions, 
        float *voxel_positions_gpu; 
        cudaMalloc((void**)&voxel_positions_gpu, 3*nvoxels*sizeof(float));
        cudaMemcpy(voxel_positions_gpu, voxel_positions.data(), 3*nvoxels*sizeof(float), cudaMemcpyHostToDevice); 
        /// float *t0, 
        float *t0_gpu;
        cudaMalloc((void**)&t0_gpu, sizeof(uint));
        cudaMemcpy(t0_gpu, &T, sizeof(uint), cudaMemcpyHostToDevice);
        /// float *deltaT,
        float *deltaT_gpu;
        cudaMalloc((void**)&deltaT_gpu, sizeof(float));
        cudaMemcpy(deltaT_gpu, &deltaT, sizeof(float), cudaMemcpyHostToDevice);
        /// uint *voxels_per_side,
        uint *voxels_per_side_gpu;
        cudaMalloc((void**)&voxels_per_side_gpu, sizeof(uint));
        cudaMemcpy(voxels_per_side_gpu, &voxels_per_side, sizeof(uint), cudaMemcpyHostToDevice);
        /// bool *is_confocal)
        bool *is_confocal_gpu;
        cudaMalloc((void**)&is_confocal_gpu, sizeof(bool));
        cudaMemcpy(is_confocal_gpu, &is_confocal, sizeof(bool), cudaMemcpyHostToDevice);
        
        dim3 dimBlock(voxels_per_side, voxels_per_side, voxels_per_side);
        dim3 dimThreads(min(camera_grid_points[0], MAX_THREADS_PER_BLOCK), min(camera_grid_points[1], MAX_THREADS_PER_BLOCK));
        cuda_backprojection<<<dimBlock, dimThreads>>>(transient_data_gpu,
                                             T_gpu,
                                             laser_grid_points_gpu,
                                             camera_grid_points_gpu,
                                             camera_grid_positions_gpu,
                                             laser_grid_positions_gpu,
                                             camera_pos_gpu,
                                             laser_pos_gpu,
                                             voxel_volume_gpu,
                                             voxel_positions_gpu,
                                             t0_gpu,
                                             deltaT_gpu,
                                             voxels_per_side_gpu,
                                             is_confocal_gpu);
        cudaDeviceSynchronize();

        // check for error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        cudaMemcpy(voxel_volume.data(), voxel_volume_gpu, nvoxels*sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(transient_data_gpu);
        cudaFree(T_gpu);
        cudaFree(laser_grid_points_gpu);
        cudaFree(camera_grid_points_gpu);
        cudaFree(camera_grid_positions_gpu);
        cudaFree(laser_grid_positions_gpu);
        cudaFree(camera_pos_gpu);
        cudaFree(laser_pos_gpu);
        cudaFree(voxel_volume_gpu);
        cudaFree(voxel_positions_gpu);
        cudaFree(t0_gpu);
        cudaFree(deltaT_gpu);
        cudaFree(voxels_per_side_gpu);
        cudaFree(is_confocal_gpu);

        return voxel_volume;
    }
}