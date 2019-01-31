#ifndef BACKPROJECT_HPP
#define BACKPROJECT_HPP

#define USE_XTENSOR
#define XTENSOR_ENABLE_XSIMD

#include <math.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-io/xhighfive.hpp>
#include <omp.h>

typedef unsigned int uint32_t;

namespace bp
{
using vector3 = xt::xtensor_fixed<float, xt::xshape<3>>;

template <typename V1, typename V2>
float distance(const V1 &p1, const V2 &p2)
{
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
    const xt::xtensor<float, 3> &camera_grid_positions,
    const xt::xtensor<float, 3> &laser_grid_positions,
    vector3 camera_position,
    vector3 laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    vector3 volume_position,
    float volume_size,
    uint32_t voxels_per_side)
{
    float voxel_size = volume_size / (voxels_per_side - 1);

    // Instead of creating a tensor with the actual 3d points, we can just compute them on the fly.
    // This may be slower or faster, depending on how expensive memory access to such a volume was (may need testing)
    auto get_point = [volume_position, volume_size, voxel_size](uint32_t x, uint32_t y, uint32_t z) -> vector3 {
        static auto zero_pos = volume_position - volume_size / 2;
        return zero_pos + vector3{x * voxel_size, y * voxel_size, z * voxel_size};
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
    for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
    {
        for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
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
        for (uint32_t lx = 0; lx < camera_grid_points[0]; lx++)
        {
            for (uint32_t ly = 0; ly < camera_grid_points[1]; ly++)
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
    for (uint32_t x = 0; x < voxels_per_side; x++)
    {
        for (uint32_t y = 0; y < voxels_per_side; y++)
        {
            for (uint32_t z = 0; z < voxels_per_side; z++)
            {
                float radiance_sum = 0.0;
                vector3 voxel_position = get_point(x, y, z);
                for (uint32_t lx = 0; lx < laser_grid_points[0]; lx++)
                {
                    for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
                    {
                        vector3 laser_wall_point = xt::view(laser_grid_positions, xt::all(), ly, lx);
                        float laser_wall_distance = laser_wall_distances(lx, ly);
                        if (!is_confocal)
                        {
                            for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
                            {
                                for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
                                {
                                    vector3 camera_wall_point = xt::view(camera_grid_positions, xt::all(), cy, cx);
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
} // namespace bp

#endif
