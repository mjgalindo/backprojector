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

struct ppd 
{
    std::array<float, 3> camera, laser;
    float camera_wall, laser_wall;
};

class iter3D 
{
private:
    size_t t_x = 0ul, t_y = 0ul, t_z = 0ul;
    size_t t_length_x, t_length_y, t_length_z, t_total_length;
    size_t t_current = 0ul;

public:
    inline size_t current() { return t_current; } 
    inline size_t total_length() { return t_total_length; }
    inline size_t x() { return t_x; }
    inline size_t y() { return t_y; }
    inline size_t z() { return t_z; }

    iter3D(size_t length_x, size_t length_y, size_t length_z) : 
        t_length_x(length_x), t_length_y(length_y), t_length_z(length_z), 
        t_total_length(length_x * length_y * length_z + length_y * length_z + length_z) {}
    
    template <class D>
    iter3D(D lengths) :
        t_length_x(lengths[0]), t_length_y(lengths[1]), t_length_z(lengths[2]), 
        t_total_length(lengths[0] * lengths[1] * lengths[2]) {}

    void operator++()
    {
        t_current++;
        if (++t_z == t_length_z)
        {
            t_z = 0ul;
            if (++t_y == t_length_y)
            {
                t_y = 0ul;
                ++t_x;
                // Bounds check for x need to be handled by the user
            }
        }
    }

    void jump_to(size_t id)
    {
        t_z = id % t_length_z;
        id = id / t_length_z;
        t_y = id % t_length_y;
        id = id / t_length_y;
        t_x = id;
    }
};

void classic_backprojection(const xt::xarray<float>& transient_data, 
                            const std::vector<ppd>& point_pairs,
                            const xt::xarray<float>& volume_position,
                            const xt::xarray<float>& volume_size,
                            const xt::xarray<float>& voxel_size,
                            xt::xarray<float>& volume,
                            float t0, float deltaT, uint32_t T)
{
    std::array<size_t, 3> voxels_per_side {volume.shape()[0], volume.shape()[1], volume.shape()[2]};

    // Instead of creating a tensor with the actual 3d points, we can just compute them on the fly.
    // This may be slower or faster, depending on how expensive memory access to such a volume was (may need testing)
    auto get_point = [volume_position, volume_size, voxel_size](uint32_t x, uint32_t y, uint32_t z) -> xt::xarray<float> {
        static auto zero_pos = volume_position - volume_size / 2;
        return std::move(zero_pos + xt::xarray<float>{x * voxel_size[0], y * voxel_size[1], z * voxel_size[2]});
    };

    int iters = 0;
    std::cout << '\r' << 0 << '/' << voxels_per_side[0] << std::flush;

    // Collapse blocks. Can't be done as is due to the progress bar
    #pragma omp parallel
    {
        // Prepare the parallel range for each thread
        uint32_t threadId = omp_get_thread_num();
        uint32_t nthreads = omp_get_num_threads();

        iter3D iter(voxels_per_side);
        float thread_iterations = iter.total_length() / (float) nthreads;
        uint32_t from = std::floor(threadId * thread_iterations);
        // If the length is not divisible by the number of threads, 
        // the last thread gets less work
        uint32_t to = std::min({(size_t) (from + std::ceil(thread_iterations)), iter.total_length()});
        
        iter.jump_to(from);
        for (int id = from; id < to; ++id)
        {
            float radiance_sum = 0.0f;
            for (int pairId = 0; pairId < point_pairs.size(); pairId++)
            {
                const auto& pair = point_pairs[pairId];
                const xt::xarray<float> voxel_position = get_point(iter.x(), iter.y(), iter.z());
                const float wall_voxel_wall_distance = distance(pair.laser, voxel_position) +
                                                       distance(voxel_position, pair.camera);
                float total_distance = pair.laser_wall + wall_voxel_wall_distance + pair.camera_wall;
                int time_index = round((total_distance - t0) / deltaT);
                if (time_index >= 0 && time_index < T)
                {
                    radiance_sum += transient_data(pairId, time_index);
                }
            }
            volume(iter.x(), iter.y(), iter.z()) = radiance_sum;
            ++iter;
            if (threadId == 0 && id % voxels_per_side[1] * voxels_per_side[2] == 0)
            {
                uint32_t slices_done = (nthreads * id) / (voxels_per_side[1] * voxels_per_side[2]);
                slices_done = slices_done > voxels_per_side[0] ? voxels_per_side[0] : slices_done;
                std::cout << '\r' << slices_done << '/' << voxels_per_side[0] << std::flush;
            }
        }
    }
    std::cout << '\r' << voxels_per_side[0] << '/' << voxels_per_side[0] << std::endl;
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
    const xt::xarray<float> volume_size,
    const xt::xarray<uint32_t> voxels_per_side)
{
    xt::xarray<float> voxel_size = volume_size / (voxels_per_side - 1);

    std::array<uint32_t, 2> camera_grid_points;
    std::array<uint32_t, 2> laser_grid_points;
    {
        // camera_grid_positions.shape() is {3, point_y, points_x}
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[0];
        camera_grid_points[1] = t[1];

        // laser_grid_positions.shape() is {3, point_y, points_x}
        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[0];
        laser_grid_points[1] = t[1];
    }

    uint32_t number_of_pairs = is_confocal ? 
        camera_grid_points[0] * camera_grid_points[1] : 
        camera_grid_points[0] * camera_grid_points[1] * 
        laser_grid_points[0] * laser_grid_points[1];

    std::vector<ppd> point_pairs(number_of_pairs);

    // Calculate distances
    uint32_t p = 0;
    if (!is_confocal)
    {
        #pragma omp parallel for collapse(2)
        for (size_t lx = 0; lx < laser_grid_points[0]; lx++)
        {
            for (size_t ly = 0; ly < laser_grid_points[1]; ly++)
            {
                xt::xarray<float> laser_point = xt::view(laser_grid_positions, lx, ly, xt::all());
                float laser_wall = distance(laser_position, laser_point);
                for (size_t cx = 0; cx <  camera_grid_points[0]; cx++)
                {
                    for (size_t cy = 0; cy <  camera_grid_points[1]; cy++)
                    {
                        auto camera_point = xt::view(camera_grid_positions, cx, cy, xt::all());
                        std::copy(laser_point.begin(), laser_point.end(), point_pairs[p].laser.begin());
                        std::copy(camera_point.begin(), camera_point.end(), point_pairs[p].camera.begin());
                        point_pairs[p].camera_wall = distance(camera_position, camera_point);
                        point_pairs[p].laser_wall = laser_wall;
                        p++;
                    }
                }
            }
        }
    }
    else
    {
        // Calculate laser-wall distances
        #pragma omp parallel for collapse(2)
        for (size_t lx = 0; lx < laser_grid_points[0]; lx++)
        {
            for (size_t ly = 0; ly < laser_grid_points[1]; ly++)
            {
                auto camera_point = xt::view(camera_grid_positions, lx, ly, xt::all());
                std::copy(camera_point.begin(), camera_point.end(), point_pairs[p].laser.begin());
                std::copy(camera_point.begin(), camera_point.end(), point_pairs[p].camera.begin());
                float laser_camera_wall = distance(camera_position, camera_point);
                point_pairs[p].camera_wall = laser_camera_wall;
                point_pairs[p].laser_wall = laser_camera_wall;
                p++;            
            }
        }
    }

    int T = transient_data.shape()[transient_data.shape().size()-1];
    xt::xarray<float> volume = xt::zeros<float>(voxels_per_side);

    classic_backprojection(transient_data, 
                           point_pairs,
                           volume_position, volume_size, 
                           voxel_size, volume,
                           t0, deltaT, T);
    
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
    bool assume_row_major = false)
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
        float voxel_volume_diagonal = sqrt(sqrt(2 * (max_size * max_size)) + max_size * max_size);
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
        max_T_index = std::min((int)ceil(max_distance / deltaT) + 200, (int)T);
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

    xt::xarray<float> volume_zero_pos = volume_position - volume_size / 2;
    xt::xarray<float> voxel_inc = volume_size / (voxels_per_side - 1);
    for (uint32_t i = 0; i < 3; i++)
        if (voxels_per_side[i] == 1)
            voxel_inc[i] = volume_size[i];

    // Copy all the necessary information to the device

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
    t0 = t0 + min_T_index * deltaT;
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
                             deltaT);

    return voxel_volume;
}

} // namespace bp

#endif
