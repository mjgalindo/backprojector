#ifndef BACKPROJECT_HPP
#define BACKPROJECT_HPP

#define USE_XTENSOR
#define XTENSOR_ENABLE_XSIMD

#include <math.h>
#include <xtensor/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xview.hpp>
#include <omp.h>

#include "backproject_cuda.hpp"

namespace bp
{

template <typename V1, typename V2>
float distance(const V1 &p1, const V2 &p2)
{
    if (p1.size() < 3 || p2.size() < 3) return 0.0;
    std::array<float, 3> tmp;
    for (int i = 0; i < 3; i++)
    {
        tmp[i] = p1[i] - p2[i];
        tmp[i] = tmp[i] * tmp[i];
    }

    return sqrt(tmp[0] + tmp[1] + tmp[2]);
}

inline double round(double x)
{
    return floor(x + 0.5);
}

xt::xarray<float> backproject(
    const xt::xarray<float> &transient_data,
    const xt::xarray<float> &camera_grid_positions,
    const xt::xarray<float> &laser_grid_positions,
    const xt::xarray<float> &camera_position,
    const xt::xarray<float> &laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    const xt::xarray<float> &volume_position,
    float volume_size,
    uint32_t voxels_per_side)
{
    float voxel_size = volume_size / (voxels_per_side - 1);

    // Instead of creating a tensor with the actual 3d points, we can just compute them on the fly.
    // This may be slower or faster, depending on how expensive memory access to such a volume was (may need testing)
    auto get_point = [volume_position, volume_size, voxel_size](uint32_t x, uint32_t y, uint32_t z) -> xt::xarray<float> {
        static auto zero_pos = volume_position - volume_size / 2;
        return zero_pos + xt::xarray<float>{x * voxel_size, y * voxel_size, z * voxel_size};
    };

    std::array<uint32_t, 2> camera_grid_points;
    std::array<uint32_t, 2> laser_grid_points;
    {
        // camera_grid_positions.shape() is {3, point_y, points_x}
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[2];
        camera_grid_points[1] = t[1];

        // laser_grid_positions.shape() is {3, point_y, points_x}
        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[2];
        laser_grid_points[1] = t[1];
    }

    xt::xarray<float> camera_wall_distances = xt::zeros<float>({camera_grid_points[0], camera_grid_points[1]});
    xt::xarray<float> laser_wall_distances = xt::zeros<float>({camera_grid_points[0], camera_grid_points[1]});

// Calculate camera-wall distances
#pragma omp parallel for
    for (int32_t cx = 0; cx < (int)camera_grid_points[0]; cx++)
    {
        for (int32_t cy = 0; cy < (int)camera_grid_points[1]; cy++)
        {
            auto wall_point = xt::view(camera_grid_positions, xt::all(), cy, cx);
            camera_wall_distances(cx, cy) = distance(camera_position, wall_point);
            if (is_confocal)
            {
                laser_wall_distances(cx, cy) = distance(laser_position, wall_point);
            }
        }
    }

    if (!is_confocal)
    {
        // Calculate laser-wall distances
#pragma omp parallel for
        for (int32_t lx = 0; lx < (int)camera_grid_points[0]; lx++)
        {
            for (int32_t ly = 0; ly < (int)camera_grid_points[1]; ly++)
            {
                auto wall_point = xt::view(camera_grid_positions, xt::all(), ly, lx);
                laser_wall_distances(lx, ly) = distance(laser_position, wall_point);
            }
        }
    }

    int T = transient_data.shape()[0];
    xt::xarray<float> volume = xt::zeros<float>({voxels_per_side, voxels_per_side, voxels_per_side});

    int iters = 0;
    std::cout << '\r' << 0 << '/' << voxels_per_side << std::flush;

#pragma omp parallel for schedule(static)
    for (int32_t x = 0; x < voxels_per_side; x++)
    {
        for (uint32_t y = 0; y < voxels_per_side; y++)
        {
            for (uint32_t z = 0; z < voxels_per_side; z++)
            {
                float radiance_sum = 0.0;
                xt::xarray<float> voxel_position = get_point(x, y, z);
                for (int32_t lx = 0; lx < (int)laser_grid_points[0]; lx++)
                {
                    for (int32_t ly = 0; ly < (int)laser_grid_points[1]; ly++)
                    {
                        xt::xarray<float> laser_wall_point = xt::view(laser_grid_positions, xt::all(), ly, lx);
                        float laser_wall_distance = laser_wall_distances(lx, ly);
                        if (!is_confocal)
                        {
                            for (int32_t cx = 0; cx < (int)camera_grid_points[0]; cx++)
                            {
                                for (int32_t cy = 0; cy < (int)camera_grid_points[1]; cy++)
                                {
                                    xt::xarray<float> camera_wall_point = xt::view(camera_grid_positions, xt::all(), cy, cx);
                                    float camera_wall_distance = camera_wall_distances(cx, cy);

                                    float wall_voxel_wall_distance = distance(laser_wall_point, voxel_position) +
                                                                     distance(voxel_position, camera_wall_point);
                                    float total_distance = laser_wall_distance + wall_voxel_wall_distance + camera_wall_distance;
                                    int time_index = round((total_distance - t0) / deltaT);
                                    if (time_index >= 0 && time_index < T)
                                    {
                                        radiance_sum += transient_data(time_index, cy, cx, ly, lx);
                                    }
                                }
                            }
                        }
                        else
                        {
                            float wall_voxel_wall_distance = distance(laser_wall_point, voxel_position) +
                                                             distance(voxel_position, laser_wall_point);
                            float total_distance = laser_wall_distance + wall_voxel_wall_distance + laser_wall_distance;
                            int time_index = round((total_distance - t0) / deltaT);
                            if (time_index >= 0 && time_index < T)
                            {
                                radiance_sum += transient_data(time_index, ly, lx);
                            }
                        }
                    }
                }
                volume(x, y, z) = radiance_sum;
            }
        }
        if (omp_get_thread_num() == 0)
        {
            uint32_t nthreads = omp_get_num_threads();
            uint32_t slices_done = (++iters) * nthreads;
            slices_done = slices_done > voxels_per_side ? voxels_per_side : slices_done;
            std::cout << '\r' << slices_done << '/' << voxels_per_side << std::flush;
        }
    }
    return volume;
}

xt::xarray<float> gpu_backproject(
    const xt::xarray<float> &transient_data,
    const xt::xarray<float> &camera_grid_positions,
    const xt::xarray<float> &laser_grid_positions,
    const xt::xarray<float> &camera_position,
    const xt::xarray<float> &laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    const xt::xarray<float> &volume_position,
    const xt::xarray<float> &volume_size,
    xt::xarray<uint32_t> voxels_per_side,
    bool assume_row_major = false,
    const xt::xarray<float> wall_normal=xt::xarray<float>{0,-1,0})
{
    auto tdata_shape = transient_data.shape();
    uint32_t T = tdata_shape[assume_row_major ? tdata_shape.size() - 1 : 0];

    std::array<uint32_t, 2> camera_grid_points;
    std::array<uint32_t, 2> laser_grid_points;
    
    {
        // camera_grid_positions.shape() is (points_x, points_y, 3)
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[assume_row_major ? 0 : 2];
        camera_grid_points[1] = t[assume_row_major ? 1 : 1];

        // laser_grid_positions.shape() is (points_x, points_y, 3)
        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[assume_row_major ? 0 : 2];
        laser_grid_points[1] = t[assume_row_major ? 1 : 1];
    }
    

    const uint32_t num_laser_points = laser_grid_points[0] * laser_grid_points[1];
    const uint32_t num_camera_points = camera_grid_points[0] * camera_grid_points[1];
    const uint32_t num_non_nan_laser_points = xt::sum(!xt::isnan(xt::view(laser_grid_positions, xt::all(), xt::all(), 0)))[0];
    const uint32_t num_non_nan_camera_points = xt::sum(!xt::isnan(xt::view(camera_grid_positions, xt::all(), xt::all(), 0)))[0];

    std::vector<pointpair> scanned_pairs;
    
    std::vector<uint32_t> sum_plane;
    if (assume_row_major)
        sum_plane = {0, 1};
    else
        sum_plane = {1, 2};
    
    xt::xarray<float> laser_grid_center = xt::nansum(laser_grid_positions, sum_plane) / num_non_nan_laser_points;
    xt::xarray<float> camera_grid_center = xt::nansum(camera_grid_positions, sum_plane) / num_non_nan_camera_points;

    // Get the minimum and maximum distance traveled to transfer as little memory to the GPU as possible
    int min_T_index = 0, max_T_index = 0;
    {
        xt::xarray<float> las_min_point;
        xt::xarray<float> las_max_point;
        xt::xarray<float> cam_min_point;
        xt::xarray<float> cam_max_point;
        if (assume_row_major)
        {
            las_min_point = xt::view(laser_grid_positions, 0, 0, xt::all());
            las_max_point = xt::view(laser_grid_positions, laser_grid_points[0] - 1, laser_grid_points[1] - 1, xt::all());
            cam_min_point = xt::view(camera_grid_positions, 0, 0, xt::all());
            cam_max_point = xt::view(camera_grid_positions, camera_grid_points[0] - 1, camera_grid_points[1] - 1, xt::all());
        }
        else
        {
            las_min_point = xt::view(laser_grid_positions, xt::all(), 0, 0);
            las_max_point = xt::view(laser_grid_positions, xt::all(), laser_grid_points[1] - 1, laser_grid_points[0] - 1);
            cam_min_point = xt::view(camera_grid_positions, xt::all(), 0, 0);
            cam_max_point = xt::view(camera_grid_positions, xt::all(), camera_grid_points[1] - 1, camera_grid_points[0] - 1);
        }
        float laser_grid_diagonal = distance(las_min_point, las_max_point);
        float camera_grid_diagonal = distance(cam_min_point, cam_max_point);
        float max_size = xt::amax(volume_size)[0];
        float voxel_volume_diagonal = sqrt(3 * max_size * max_size);
        float min_distance = abs(distance(laser_position, laser_grid_center) - laser_grid_diagonal / 2 +
                             2 * (distance(laser_grid_center, volume_position) - voxel_volume_diagonal / 2) +
                             distance(camera_position, camera_grid_center) - camera_grid_diagonal / 2);
        float max_distance = abs(distance(laser_position, laser_grid_center) + laser_grid_diagonal / 2 +
                             2 * (distance(laser_grid_center, volume_position) + voxel_volume_diagonal / 2) +
                             distance(camera_position, camera_grid_center) + camera_grid_diagonal / 2);
        // Adjust distances by the minimum recorded distance.
        min_distance -= t0;
        max_distance -= t0;
        min_T_index = std::max((int)floor(min_distance / deltaT) - 100, 0);
        max_T_index = std::min((int)ceil(max_distance / deltaT) + 100, (int)T);
    }

    // Gather the captured pairs into a single vector to pass to the GPU.
    if (is_confocal)
    {
        scanned_pairs.resize(num_laser_points);
#pragma omp parallel for
        for (int32_t cx = 0; cx < camera_grid_points[0]; cx++)
            for (int32_t cy = 0; cy < camera_grid_points[1]; cy++)
            {
                uint32_t index = cx * camera_grid_points[1] + cy;
                for (int32_t i = 0; i < 3; i++)
                {
                    if (assume_row_major)
                    {
                        scanned_pairs[index].cam_point[i] = camera_grid_positions(cx, cy, i);
                        scanned_pairs[index].laser_point[i] = camera_grid_positions(cx, cy, i);
                    }
                    else
                    {
                        scanned_pairs[index].cam_point[i] = camera_grid_positions(i, cy, cx);
                        scanned_pairs[index].laser_point[i] = camera_grid_positions(i, cy, cx);
                    }
                }
            }
    }
    else
    {
        scanned_pairs.resize(num_laser_points *
                             num_camera_points);

#pragma omp parallel for
        for (int32_t lx = 0; lx < laser_grid_points[0]; lx++)
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
                            if (assume_row_major)
                            {
                                scanned_pairs[index].cam_point[i] = camera_grid_positions(cx, cy, i);
                                scanned_pairs[index].laser_point[i] = laser_grid_positions(lx, ly, i);
                            }
                            else
                            {
                                scanned_pairs[index].cam_point[i] = camera_grid_positions(i, cy, cx);
                                scanned_pairs[index].laser_point[i] = laser_grid_positions(i, ly, lx);
                            }
                        }
                    }
    }
    
    // Compute the reconstruction volume so that it is aligned to the relay-wall
    xt::xarray<float> x({1,0,0});
    xt::xarray<float> y({0,1,0});
    xt::xarray<float> z({0,0,1});

    if (wall_normal[0] != 0 || wall_normal[1] != 0 || wall_normal[2] != 0)
    {   
        y = wall_normal;
        x = xt::linalg::cross(y,z);
        z = xt::linalg::cross(y,x);
        x = x / xt::sqrt(xt::sum(x*x));
        y = y / xt::sqrt(xt::sum(y*y));
        z = z / xt::sqrt(xt::sum(z*z));
    }

    xt::xarray<float> volume_zero_pos = volume_position - (volume_size * x + volume_size * y + volume_size * z)  / 2;
    xt::xarray<float> voxel_inc = xt::stack(xt::xtuple(
                                    (x * volume_size[0]) / (voxels_per_side[0] - 1), 
                                    (y * volume_size[1]) / (voxels_per_side[1] - 1),
                                    (z * volume_size[2]) / (voxels_per_side[2] - 1)));
                                    
    /// float *transient_data,
    if (assume_row_major)
    {
        std::cout << "Copying compact transient data measurements" << std::flush;
    }
    else
    {
        std::cout << "Copying compact transposed transient data measurements" << std::flush;
    }

    xt::xarray<float> transient_chunk;
    // TODO: Fill with zeroes if max_T_index > T
    if (is_confocal)
    {
        if (assume_row_major)
        {
            transient_chunk = xt::view(transient_data, xt::all(), xt::all(), xt::range(min_T_index, max_T_index));
        }
        else
        {
            transient_chunk = xt::empty<float>({laser_grid_points[0], laser_grid_points[1], (uint32_t)(max_T_index - min_T_index)});
#pragma omp parallel for
            for (int32_t lx = 0; lx < laser_grid_points[0]; lx++)
                for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
                {
                    xt::view(transient_chunk, lx, ly, xt::all()) = xt::view(transient_data, xt::range(min_T_index, max_T_index), ly, lx);
                }
        }
    }
    else
    {
        if (assume_row_major)
        {
            transient_chunk = xt::view(transient_data, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(min_T_index, max_T_index));
        }
        else
        {
            transient_chunk = xt::empty<float>({laser_grid_points[0], laser_grid_points[1], camera_grid_points[0], camera_grid_points[1], (uint32_t)(max_T_index - min_T_index)});
#pragma omp parallel for
            for (uint32_t lx = 0; lx < laser_grid_points[0]; lx++)
                for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
                    for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
                        for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
                        {
                            xt::view(transient_chunk, lx, ly, cx, cy, xt::all()) = xt::view(transient_data, xt::range(min_T_index, max_T_index), cy, cx, ly, lx);
                        }
        }
    }
    
    transient_chunk = xt::nan_to_num(transient_chunk);

    uint32_t total_transient_size = 1;
    for (const auto &d : transient_chunk.shape())
    {
        total_transient_size *= d;
    }
    t0 = t0 + ((float) min_T_index) * deltaT;
    std::cout << " Done!" << std::endl;

    uint32_t chunkedT = (uint32_t)(max_T_index - min_T_index);
    xt::xarray<float> voxel_volume = xt::zeros<float>(voxels_per_side);
    call_cuda_backprojection(transient_chunk.data(),
                             total_transient_size,
                             chunkedT,
                             scanned_pairs,
                             camera_position.size() > 1 ? camera_position.data() : nullptr,
                             laser_position.size() > 1 ? laser_position.data() : nullptr,
                             voxel_volume.data(),
                             voxels_per_side.data(),
                             volume_zero_pos.data(),
                             voxel_inc.data(),
                             t0,
                             deltaT, true);

    return voxel_volume;
}

} // namespace bp

#endif
