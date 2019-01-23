
#include <chrono>
#include <math.h>
#include <helper_cuda.h>
#include <vector>

#include "backproject_cuda.hpp"

namespace bp_cuda
{
const uint MAX_THREADS_PER_BLOCK = 1;

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

struct pointpair{
    float cam_point[3], laser_point[3];
};

__device__
void ppair(const pointpair* pp)
{
    printf("CP: [%.2f,%.2f,%.2f], [%.2f,%.2f,%.2f]\n", 
        pp->cam_point[0], pp->cam_point[1], pp->cam_point[2],
        pp->laser_point[0], pp->laser_point[1], pp->laser_point[2]);
}

__global__
void cuda_backprojection(float *transient_data,
                         uint *T,
                         uint *num_pairs,
                         pointpair *scanned_pairs,
                         float *camera_pos,
                         float *laser_pos,
                         float *voxel_volume,
                         float *volume_zero_pos,
                         float *voxel_inc,
                         float *t0,
                         float *deltaT,
                         uint *voxels_per_side)
{
    printf("Started block %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);

    uint voxel_id = blockIdx.x * voxels_per_side[0] * voxels_per_side[1] + blockIdx.y * voxels_per_side[2] + blockIdx.z;
    __shared__ double local_array[MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK];
    double& radiance_sum = local_array[threadIdx.x*MAX_THREADS_PER_BLOCK];
    radiance_sum = 0.0;

    float voxel_position[] = {volume_zero_pos[0]+voxel_inc[0]*blockIdx.x,
                              volume_zero_pos[1]+voxel_inc[1]*blockIdx.y,
                              volume_zero_pos[2]+voxel_inc[2]*blockIdx.z};

    //printf("VOXEL: %.2f, %.2f, %.2f\n", voxel_position[0], voxel_position[1], voxel_position[2]);
    for (uint i = 0; i < *num_pairs / MAX_THREADS_PER_BLOCK; i++)
    {
        uint pair_index = i * MAX_THREADS_PER_BLOCK + threadIdx.x;
        //printf("Block %d %d %d loading pair %d\n", blockIdx.x, blockIdx.y, blockIdx.z, pair_index);
        const pointpair& pair = scanned_pairs[pair_index];
        if (blockIdx.x == 0 && blockIdx.y == 0 && (pair_index > 65500)) ppair(&pair);
        // From the laser to the wall
        float laser_wall_distance = distance(laser_pos, pair.laser_point);
        // From the wall to the current voxel
        float laser_point_voxel_distance = distance(pair.laser_point, voxel_position);
        
        // From the wall back to the camera
        float cam_wall_distance = distance(pair.cam_point, camera_pos);
        // From the object back to the wall
        float voxel_cam_point_distance = distance(voxel_position, pair.cam_point);

        // Radiance gets attenuated with the square of traveled distance between bounces
        float distance_attenuation = laser_wall_distance * laser_wall_distance +
                                     laser_point_voxel_distance * laser_point_voxel_distance +
                                     voxel_cam_point_distance * voxel_cam_point_distance +
                                     cam_wall_distance * cam_wall_distance;
        
        // TODO: Cosine attenuation due to Lambert's law

        float total_distance = laser_wall_distance + laser_point_voxel_distance + voxel_cam_point_distance + cam_wall_distance;
        
        uint time_index = round((total_distance - *t0) / *deltaT);
        if (time_index < *T)
        {   
            uint tdindex = pair_index * *T + time_index;
            radiance_sum += transient_data[tdindex] * distance_attenuation;
        }
        // if (blockIdx.x == 0 && blockIdx.y == 0 && (pair_index > 65500)) printf("xyz: %d-%d-%d, I: %d, time_index %d, TDINDEX: %d MAX: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, pair_index, time_index, pair_index * *T + time_index, *num_pairs * *T);
    }

    __syncthreads();
    float total_radiance = 0.0;
    for (int i = 0; i < MAX_THREADS_PER_BLOCK*MAX_THREADS_PER_BLOCK; i++) {
        total_radiance += local_array[i];
    }

    // All threads write the same value
    printf("ABOUT TO WRITE %d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    voxel_volume[voxel_id] = total_radiance;
    printf("WRITTEN\n");
}


xt::xarray<float> backproject(
    xt::xarray<float>& transient_data,
    const xt::xtensor<float, 3>& camera_grid_positions,
    const xt::xtensor<float, 3>& laser_grid_positions,
    vector3 camera_position,
    vector3 laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    vector3 volume_position,
    float volume_size,
    uint voxels_per_side)
{
    std::array<uint, 2> camera_grid_points;
    std::array<uint, 2> laser_grid_points;
    {
        // camera_grid_positions.shape() -> (points_x, points_y, 3)
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[2];
        camera_grid_points[1] = t[1];

        // laser_grid_positions.shape() -> (points_x, points_y, 3)
        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[2];
        laser_grid_points[1] = t[1];
    }

    std::vector<pointpair> scanned_pairs;

    if (is_confocal) {
        scanned_pairs.resize(laser_grid_points[0] * laser_grid_points[1]);
    }
    else 
    {
        scanned_pairs.resize(laser_grid_points[0] * laser_grid_points[1] * 
                              camera_grid_points[0] * camera_grid_points[1]);
        
        #pragma omp parallel for
        for (uint lx = 0; lx < laser_grid_points[0]; lx++)
        for (uint ly = 0; ly < laser_grid_points[1]; ly++)
        for (uint cx = 0; cx < camera_grid_points[0]; cx++)
        for (uint cy = 0; cy < camera_grid_points[1]; cy++)
        {
            uint index = lx * laser_grid_points[1] * camera_grid_points[0] * camera_grid_points[1] +
                         ly * camera_grid_points[0] * camera_grid_points[1] +
                         cx * camera_grid_points[1] +
                         cy;
            for (int i = 0; i < 3; i++)
            {
                scanned_pairs[index].cam_point[i] = camera_grid_positions(i, cy, cx);
                scanned_pairs[index].laser_point[i] = laser_grid_positions(i, ly, lx);
            }
        }
    }
    
    uint num_pairs = scanned_pairs.size();
    std::cout << num_pairs << std::endl;

    uint T = transient_data.shape()[0];
    uint total_transient_size = sizeof(float);
    for (const auto d : transient_data.shape()) {
        total_transient_size *= d;
    }

    vector3 volume_zero_pos = volume_position - volume_size / 2;
    float voxel_size = volume_size / (voxels_per_side - 1);
    if (voxels_per_side == 1) voxel_size = volume_size;
    vector3 voxel_inc{voxel_size, voxel_size, voxel_size};

    // Copy all the necessary information to the device

    /// float *transient_data,
    float *transient_data_gpu;
    std::cout << "NOT Transposing measurements" << std::flush;

    // transient_data = xt::transpose(transient_data);
    std::cout << " Done!" << std::endl;
    cudaMalloc((void**)&transient_data_gpu, total_transient_size);
    cudaMemcpy(transient_data_gpu, transient_data.data(), total_transient_size, cudaMemcpyHostToDevice); 
    /// uint *T,
    uint *T_gpu;
    cudaMalloc((void**)&T_gpu, sizeof(uint));
    cudaMemcpy(T_gpu, &T, sizeof(uint), cudaMemcpyHostToDevice);
    /// uint *num_pairs
    uint *num_pairs_gpu;
    cudaMalloc((void**)&num_pairs_gpu, sizeof(uint));
    cudaMemcpy(num_pairs_gpu, &num_pairs, sizeof(uint), cudaMemcpyHostToDevice);
    /// pointpair *scanned_pairs,
    pointpair* scanned_pairs_gpu;
    cudaMalloc((void**)&scanned_pairs_gpu, num_pairs*sizeof(pointpair));
    cudaMemcpy(scanned_pairs_gpu, scanned_pairs.data(), num_pairs*sizeof(pointpair), cudaMemcpyHostToDevice);
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
    cudaMemset(voxel_volume_gpu, 0, nvoxels*sizeof(float)); 
    /// float *volume_zero_pos, 
    float *volume_zero_pos_gpu; 
    cudaMalloc((void**)&volume_zero_pos_gpu, 3*sizeof(float));
    cudaMemcpy(volume_zero_pos_gpu, volume_zero_pos.data(), 3*sizeof(float), cudaMemcpyHostToDevice); 
    /// float *voxel_inc, 
    float *voxel_inc_gpu; 
    cudaMalloc((void**)&voxel_inc_gpu, 3*sizeof(float));
    cudaMemcpy(voxel_inc_gpu, voxel_inc.data(), 3*sizeof(float), cudaMemcpyHostToDevice); 
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
    cudaMalloc((void**)&voxels_per_side_gpu, 3*sizeof(uint));
    cudaMemset(voxels_per_side_gpu, voxels_per_side, 3*sizeof(uint));
    
    dim3 dimBlock(voxels_per_side, voxels_per_side, voxels_per_side);
    dim3 dimThreads(MAX_THREADS_PER_BLOCK, 1, 1);
    cuda_backprojection<<<dimBlock, dimThreads>>>(transient_data_gpu,
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
                                                  voxels_per_side_gpu);
    cudaDeviceSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Initialize voxel volume for the CPU
    xt::xarray<float> voxel_volume = xt::zeros<float>({voxels_per_side, voxels_per_side, voxels_per_side});
    cudaMemcpy(voxel_volume.data(), voxel_volume_gpu, nvoxels*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(transient_data_gpu);
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

    return voxel_volume;
}
}