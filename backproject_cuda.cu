
#include <chrono>
#include <math.h>
#include <vector>

#include "backproject_cuda.hpp"

namespace bp_cuda
{
const uint32_t MAX_THREADS_PER_BLOCK = 32;

template <typename V1, typename V2>
float distance_cpu(const V1 &p1, const V2 &p2)
{
    std::array<float, 3> tmp;
    for (int i = 0; i < 3; i++)
    {
        tmp[i] = p1[i] - p2[i];
        tmp[i] = tmp[i] * tmp[i];
    }

    return sqrt(tmp[0] + tmp[1] + tmp[2]);
}

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

__global__
void cuda_backprojection(float *transient_data,
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
                         uint32_t *voxels_per_side)
{
    uint32_t voxel_id = blockIdx.x * voxels_per_side[0] * voxels_per_side[1] + blockIdx.y * voxels_per_side[2] + blockIdx.z;
    __shared__ double local_array[MAX_THREADS_PER_BLOCK];
    double& radiance_sum = local_array[threadIdx.x];
    radiance_sum = 0.0;

    float voxel_position[] = {volume_zero_pos[0]+voxel_inc[0]*blockIdx.x,
                              volume_zero_pos[1]+voxel_inc[1]*blockIdx.y,
                              volume_zero_pos[2]+voxel_inc[2]*blockIdx.z};

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
    uint32_t voxels_per_side)
{
    // TODO: This assumes the time dimension is first
    uint32_t T = transient_data.shape()[0];

    std::array<uint32_t, 2> camera_grid_points;
    std::array<uint32_t, 2> laser_grid_points;
    
    {
        // camera_grid_positions.shape() is (points_x, points_y, 3)
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[2];
        camera_grid_points[1] = t[1];

        // laser_grid_positions.shape() is (points_x, points_y, 3)
        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[2];
        laser_grid_points[1] = t[1];
    }

    const uint32_t num_laser_points = laser_grid_points[0] * laser_grid_points[1];
    const uint32_t num_camera_points = camera_grid_points[0] * camera_grid_points[1];

    std::vector<pointpair> scanned_pairs;

    xt::xarray<float> laser_grid_center = xt::sum(laser_grid_positions, {1,2}) / num_laser_points;
    xt::xarray<float> camera_grid_center = xt::sum(camera_grid_positions, {1,2}) / num_camera_points;

    // Get the minimum and maximum distance traveled to transfer as little memory to the GPU as possible
    int min_T_index = 0, max_T_index = 0;
    {
        float laser_grid_diagonal = distance_cpu(xt::view(laser_grid_positions, xt::all(), 0, 0), 
                                            xt::view(laser_grid_positions, xt::all(), 
                                                laser_grid_points[1]-1, laser_grid_points[0]-1));
        float camera_grid_diagonal = distance_cpu(xt::view(camera_grid_positions, xt::all(), 0, 0), 
                                            xt::view(camera_grid_positions, xt::all(), 
                                                camera_grid_points[1]-1, camera_grid_points[0]-1));
        float voxel_volume_diagonal = sqrt(2*(volume_size * volume_size));
        float min_distance = distance_cpu(laser_position, laser_grid_center) - laser_grid_diagonal / 2 +    
                            2 * (distance_cpu(laser_grid_center, volume_position) - voxel_volume_diagonal / 2) +
                            distance_cpu(camera_position, camera_grid_center) - camera_grid_diagonal / 2;
        float max_distance = distance_cpu(laser_position, laser_grid_center) + laser_grid_diagonal / 2 +    
                            2 * (distance_cpu(laser_grid_center, volume_position) + voxel_volume_diagonal / 2) +
                            distance_cpu(camera_position, camera_grid_center) + camera_grid_diagonal / 2;

        min_T_index = std::max(int(floor(min_distance / deltaT)) - 500, 0);
        max_T_index = std::min(int(ceil(max_distance / deltaT)) + 500, (int)T);
    }

    // Gather the captured pairs into a single vector to pass to the GPU.
    if (is_confocal) {
        scanned_pairs.resize(num_laser_points);
        #pragma omp parallel for collapse(2)
        for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
        for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
        {
            uint32_t index = cx * camera_grid_points[1] + cy;
            for (uint32_t i = 0; i < 3; i++)
            {
                scanned_pairs[index].cam_point[i] = camera_grid_positions(i, cy, cx);
                scanned_pairs[index].laser_point[i] = camera_grid_positions(i, cy, cx);
            }
        }
    }
    else 
    {
        scanned_pairs.resize(num_laser_points * 
                             num_camera_points);
        
        #pragma omp parallel for collapse(4)
        for (uint32_t lx = 0; lx < laser_grid_points[0]; lx++)
        for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
        for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
        for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
        {
            uint32_t index = lx * laser_grid_points[1] * num_camera_points +
                         ly * num_camera_points +
                         cx * camera_grid_points[1] +
                         cy;
            for (uint32_t i = 0; i < 3; i++)
            {
                scanned_pairs[index].cam_point[i] = camera_grid_positions(i, cy, cx);
                scanned_pairs[index].laser_point[i] = laser_grid_positions(i, ly, lx);
            }
        }
    }
    
    uint32_t num_pairs = scanned_pairs.size();

    vector3 volume_zero_pos = volume_position - volume_size / 2;
    float voxel_size = volume_size / (voxels_per_side - 1);
    if (voxels_per_side == 1) voxel_size = volume_size;
    vector3 voxel_inc{voxel_size, voxel_size, voxel_size};

    // Copy all the necessary information to the device

    /// float *transient_data,
    std::cout << "Transposing measurements" << std::flush;
    xt::xarray<float> transient_chunk;
    if (is_confocal)
    {
        transient_chunk = xt::empty<float>({laser_grid_points[0], laser_grid_points[1], (uint32_t) (max_T_index - min_T_index)});
        #pragma omp parallel for collapse(2)
        for (uint32_t lx = 0; lx < laser_grid_points[0]; lx++)
        for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
        {
            xt::view(transient_chunk, lx, ly, xt::all()) = xt::view(transient_data, xt::all(), ly, lx);
        }
    }
    else
    {
        transient_chunk = xt::empty<float>({laser_grid_points[0], laser_grid_points[1], camera_grid_points[0], camera_grid_points[1], (uint32_t) (max_T_index - min_T_index)});
        #pragma omp parallel for collapse(4)
        for (uint32_t lx = 0; lx < laser_grid_points[0]; lx++)
        for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
        for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
        for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
        {
            xt::view(transient_chunk, lx, ly, cx, cy, xt::all()) = xt::view(transient_data, xt::range(min_T_index, max_T_index), cy, cx, ly, lx);
        }
    }

    uint32_t total_transient_size = sizeof(float);
    for (const auto& d : transient_chunk.shape()) {
        total_transient_size *= d;
    }

    std::cout << " Done!" << std::endl;
    float *transient_chunk_gpu;
    cudaMalloc((void**)&transient_chunk_gpu, total_transient_size);
    cudaMemcpy(transient_chunk_gpu, transient_chunk.data(), total_transient_size, cudaMemcpyHostToDevice); 
    /// uint32_t *T,
    uint32_t *T_gpu;
    uint32_t chunkedT = (uint32_t) (max_T_index - min_T_index);
    cudaMalloc((void**)&T_gpu, sizeof(uint32_t));
    cudaMemcpy(T_gpu, &chunkedT, sizeof(uint32_t), cudaMemcpyHostToDevice);
    /// uint32_t *num_pairs
    uint32_t *num_pairs_gpu;
    cudaMalloc((void**)&num_pairs_gpu, sizeof(uint32_t));
    cudaMemcpy(num_pairs_gpu, &num_pairs, sizeof(uint32_t), cudaMemcpyHostToDevice);
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
    // Initialize voxel volume for the CPU
    xt::xarray<float> voxel_volume = xt::zeros<float>({voxels_per_side, voxels_per_side, voxels_per_side});
    float *voxel_volume_gpu;
    uint32_t nvoxels = voxels_per_side*voxels_per_side*voxels_per_side;
    cudaMalloc((void**)&voxel_volume_gpu, nvoxels*sizeof(float));
    cudaMemcpy(voxel_volume_gpu, voxel_volume.data(), nvoxels*sizeof(float), cudaMemcpyHostToDevice); 
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
    cudaMalloc((void**)&t0_gpu, sizeof(uint32_t));
    cudaMemcpy(t0_gpu, &T, sizeof(uint32_t), cudaMemcpyHostToDevice);
    /// float *deltaT,
    float *deltaT_gpu;
    cudaMalloc((void**)&deltaT_gpu, sizeof(float));
    cudaMemcpy(deltaT_gpu, &deltaT, sizeof(float), cudaMemcpyHostToDevice);
    /// uint32_t *voxels_per_side,
    uint32_t *voxels_per_side_gpu;
    cudaMalloc((void**)&voxels_per_side_gpu, 3*sizeof(uint32_t));
    {
        uint32_t tmp_vps[] = {voxels_per_side, voxels_per_side, voxels_per_side};
        cudaMemcpy(voxels_per_side_gpu, tmp_vps, 3*sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    
    dim3 dimBlock(voxels_per_side, voxels_per_side, voxels_per_side);
    dim3 dimThreads(MAX_THREADS_PER_BLOCK, 1, 1);
    std::cout << "Backprojecting on the GPU" << std::endl;
    cuda_backprojection<<<dimBlock, dimThreads>>>(transient_chunk_gpu,
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

    // check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    cudaMemcpy(voxel_volume.data(), voxel_volume_gpu, nvoxels*sizeof(float), cudaMemcpyDeviceToHost);
    
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

    return voxel_volume;
}
}