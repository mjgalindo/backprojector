#ifndef BACKPROJECT_HPP
#define BACKPROJECT_HPP

#define USE_XTENSOR
#define XTENSOR_ENABLE_XSIMD

#include <complex>
#include <math.h>
#include <xtensor/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xview.hpp>
#include <omp.h>

#include "backproject_cuda.hpp"
#include "octree_volume.hpp"

#if __linux__
#include "tqdm.h"
#endif

namespace bp
{

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

template <typename AT>
AT accumulate_radiance(const xt::xarray<AT>& transient_data,
                       const std::vector<ppd>& point_pairs,
                       const xt::xarray<float>& voxel_position,
                       float t0, float deltaT, uint32_t T, float access_width)
{
    AT radiance_sum = AT{};
    for (int pairId = 0; pairId < point_pairs.size(); pairId++)
    {
        const auto& pair = point_pairs[pairId];
        const float wall_voxel_wall_distance = distance(pair.laser_point, voxel_position) +
                                               distance(voxel_position, pair.camera_point);
        float total_distance = pair.laser_wall + wall_voxel_wall_distance + pair.camera_wall;
        int first_time_index = std::max({0, (int) round((total_distance - access_width / 2 - t0) / deltaT)});
        int last_time_index = std::min({(int) T-1, (int) round((total_distance + access_width / 2 - t0) / deltaT)});
        for (int time_index = first_time_index; time_index < last_time_index; time_index++)
        {
            radiance_sum += transient_data(pairId, time_index);
        }
    }
    return radiance_sum;
}

template <typename AT>
bool within_threshold(AT value, AT threshold)
{
    return value >= threshold;
}

template <typename AT>
bool within_threshold(std::complex<AT> value, std::complex<AT> threshold)
{
    return within_threshold(std::abs(value), std::abs(threshold));
}

template <typename AT>
void classic_backprojection(const xt::xarray<AT>& transient_data, 
                            const std::vector<ppd>& point_pairs,
                            OctreeVolumeF<AT>& volume, int depth,
                            float t0, float deltaT, uint32_t T,
                            AT threshold = 0.0f, bool verbose=true)
{
    auto max_voxels = volume.max_voxels(depth);
    xt::xarray<float> voxel_size = volume.voxel_size_at(depth);
    float voxel_diagonal = xt::sqrt(xt::sum(voxel_size * voxel_size))[0]; 
    int iters = 0;
    
    #if __linux__
    tqdm bar;
    bar.set_theme_braille();
    #else
    if (verbose)
    {
        std::cout << '\r' << 0 << '/' << max_voxels[0] << std::flush;
    }
    #endif
    int avoided = 0;

    // For loops gathered in one parallel region to get progress updates
    #pragma omp parallel shared(avoided)
    {
        // Prepare the parallel range for each thread
        uint32_t threadId = omp_get_thread_num();
        uint32_t nthreads = omp_get_num_threads();
        iter3D iter(max_voxels);
        int total_length = iter.total_length();
        float thread_iterations = total_length / (float) nthreads;
        uint32_t from = std::floor(threadId * thread_iterations);
        // If the length is not divisible by the number of threads, 
        // the last thread gets less work
        uint32_t to = from + std::ceil(thread_iterations);
        if (threadId == nthreads - 1) to = iter.total_length();
        int local_avoided = 0;
        iter.jump_to(from);
        for (int id = from; id < to; ++id)
        {
            // Skip voxels below the given threshold
            if (depth <= 0 || within_threshold(volume(iter, volume.max_depth()-1, OctreeVolumeF<AT>::BuffType::Buffer), threshold)) 
            {
                AT radiance = accumulate_radiance(transient_data, point_pairs,
                                                  volume.position_at(iter, depth),
                                                  t0, deltaT, T, voxel_diagonal);
                volume(iter, OctreeVolumeF<AT>::BuffType::Main) = radiance;
            }
            else
            {
                local_avoided++;
                volume(iter, depth) = NAN; // volume(iter, volume.max_depth()-1, SimpleOctreeVolume::BuffType::Buffer);
            }
            ++iter;

            // Update progress estimate
            if (verbose && threadId == 0)
            {
                uint32_t pairs_done = nthreads * id;
                #if __linux__
                bar.progress(pairs_done, total_length);
                #else
                std::cout << '\r' << pairs_done << '/' << total_length << std::flush;
                #endif
            }
        }
        #pragma omp atomic
        avoided += local_avoided;
    }
    if (verbose)
    {
        #if __linux__
        bar.finish();
        #else
        std::cout << '\r' << total_length << '/' << total_length << std::endl;
        #endif
        std::cout << "Avoided computing " << avoided << " blocks." << std::endl;
    }
}


template <typename AT>
void classic_backprojection(const xt::xarray<AT>& transient_data, 
                            const std::vector<ppd>& point_pairs,
                            const xt::xarray<float>& volume_size,
                            const xt::xarray<float>& volume_position,
                            xt::xarray<AT>& volume,
                            float t0, float deltaT, uint32_t T, bool verbose=true)
{
    OctreeVolumeF<AT> ov(volume, volume_size, volume_position);
    classic_backprojection(transient_data, point_pairs, ov, -1, t0, deltaT, T, 0.0f, verbose);
    volume = ov.volume();
}

template <typename AT>
void octree_backprojection(const xt::xarray<AT>& transient_data, 
                           const std::vector<ppd>& point_pairs,
                           OctreeVolumeF<AT>& volume,
                           float t0, float deltaT, uint32_t T, bool verbose=true)
{
    int depth = 3;
    AT threshold = -1.0f;
    volume.reset_buffer();
    while (depth <= volume.max_depth())
    {
        auto max_voxels = volume.max_voxels(depth);
        std::cout << "Backprojecting depth " << depth << " with threshold " << threshold << " size " << max_voxels << std::endl;
        classic_backprojection(transient_data, point_pairs, volume, depth, t0, deltaT, T, threshold, verbose);
        auto shape = xt::view(volume.volume(), 
                                      xt::range(0, max_voxels[0]), 
                                      xt::range(0, max_voxels[1]),
                                      xt::range(0, max_voxels[2])).shape();

        // xt::nan_to_num(xt::vie....
        threshold = xt::mean(xt::view(volume.volume(),
                                      xt::range(0, max_voxels[0]),
                                      xt::range(0, max_voxels[1]),
                                      xt::range(0, max_voxels[2])))[0] / (xt::mean(max_voxels)[0]/8);

        std::cout << "Volume sum " << xt::nansum(volume.volume())[0] << " " << xt::nansum(xt::view(volume.volume(), 
                                      xt::range(0, max_voxels[0]),
                                      xt::range(0, max_voxels[1]),
                                      xt::range(0, max_voxels[2]))) << std::endl;
        std::cout << "Buffer sum " << xt::nansum(volume.volume(OctreeVolumeF<AT>::BuffType::Buffer))[0] << " " << xt::nansum(xt::view(volume.volume(OctreeVolumeF<AT>::BuffType::Buffer), 
                                      xt::range(0, max_voxels[0]),
                                      xt::range(0, max_voxels[1]),
                                      xt::range(0, max_voxels[2]))) << std::endl;

        if (depth < volume.max_depth())
        {
            std::cout << "Swapping buffers" << std::endl;
            volume.swap_buffers();
        }
        depth++;
    }
}

std::vector<ppd> precompute_distances(const xt::xarray<float>& laser_position,
                                      const xt::xarray<float>& laser_grid_positions,
                                      const xt::xarray<uint32_t>& laser_grid_points,
                                      const xt::xarray<float>& camera_position,
                                      const xt::xarray<float>& camera_grid_positions,
                                      const xt::xarray<uint32_t>& camera_grid_points,
                                      bool is_confocal)
{
    uint32_t number_of_pairs = is_confocal ? 
        camera_grid_points[0] * camera_grid_points[1] : 
        camera_grid_points[0] * camera_grid_points[1] * 
        laser_grid_points[0] * laser_grid_points[1];

    std::vector<ppd> point_pairs(number_of_pairs);
    // Calculate distances
    if (!is_confocal)
    {
        #pragma omp parallel for collapse(4)
        for (size_t lx = 0; lx < laser_grid_points[0]; lx++)
        {
            for (size_t ly = 0; ly < laser_grid_points[1]; ly++)
            {
                for (size_t cx = 0; cx <  camera_grid_points[0]; cx++)
                {
                    for (size_t cy = 0; cy <  camera_grid_points[1]; cy++)
                    {
                        xt::xarray<float> laser_point = xt::view(laser_grid_positions, lx, ly, xt::all());
                        float laser_wall = distance(laser_position, laser_point);
                        auto camera_point = xt::view(camera_grid_positions, cx, cy, xt::all());
                        uint32_t p_id = lx * laser_grid_points[1] * camera_grid_points[0] * camera_grid_points[1] +
                                     ly * camera_grid_points[0] * camera_grid_points[1] +
                                     cx * camera_grid_points[1] +
                                     cy;
                        std::copy(laser_point.begin(), laser_point.end(), point_pairs[p_id].laser_point);
                        std::copy(camera_point.begin(), camera_point.end(), point_pairs[p_id].camera_point);
                        point_pairs[p_id].camera_wall = distance(camera_position, camera_point);
                        point_pairs[p_id].laser_wall = laser_wall;
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
                uint32_t p = lx * laser_grid_points[1] + ly;
                std::copy(camera_point.begin(), camera_point.end(), point_pairs[p].laser_point);
                std::copy(camera_point.begin(), camera_point.end(), point_pairs[p].camera_point);
                float laser_camera_wall = distance(camera_position, camera_point);
                point_pairs[p].camera_wall = laser_camera_wall;
                point_pairs[p].laser_wall = laser_camera_wall;
                p++;
            }
        }
    }
    return std::move(point_pairs);
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
    const xt::xarray<uint32_t> voxels_per_side,
    bool use_octree = false)
{
    xt::xarray<float> voxel_size = volume_size / (voxels_per_side - 1);

    xt::xarray<uint32_t> camera_grid_points({2});
    xt::xarray<uint32_t> laser_grid_points({2});
    {
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[0];
        camera_grid_points[1] = t[1];

        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[0];
        laser_grid_points[1] = t[1];
    }

    std::vector<ppd> point_pairs = precompute_distances(laser_position,
                                                        laser_grid_positions,
                                                        laser_grid_points,
                                                        camera_position,
                                                        camera_grid_positions,
                                                        camera_grid_points,
                                                        is_confocal);

    int T = transient_data.shape().back();

    SimpleOctreeVolume ov(voxels_per_side, volume_size, volume_position);
    if (use_octree)
    {
        octree_backprojection(transient_data, point_pairs, ov, t0, deltaT, T);
    }
    else
    {
        classic_backprojection(transient_data, point_pairs, ov, -1, t0, deltaT, T);
    }
    
    return ov.volume();
}

xt::xarray<std::complex<float>> phasor_pulse(const xt::xarray<float> &transient_data_in,
                                             float wavelength, float deltaT, uint32_t times=4)
{
    // Time bins covered by the phasor pulse
    auto pulse_size = (int) (times * wavelength / deltaT);
    // Time bins covered by a sincle cycle
    auto cycle_size = (int) (wavelength / deltaT);

    // Select a sigma so that the 99% of the gaussian is inside the pulse limits
    float vsigma = (times * wavelength) / 6;

    // Virtual emitter emission profile
    auto t = deltaT * (xt::cast<double>(xt::arange(1, pulse_size+1)) - pulse_size / 2.0);
    auto gaussian_pulse = xt::exp(-t*t / (2 * vsigma*vsigma));

    auto progression = 2 * M_PI * (1.0 / cycle_size * xt::cast<double>(xt::arange(1, pulse_size+1)));

    xt::xarray<float> sin_wave = xt::sin(progression);
    xt::xarray<float> cos_wave = xt::cos(progression);
    
    xt::xarray<float> cos_pulse = xt::squeeze(cos_wave * gaussian_pulse);
    xt::xarray<float> sin_pulse = xt::squeeze(sin_wave * gaussian_pulse);

    auto data_shape = transient_data_in.shape();
    int rows = std::accumulate(data_shape.begin(), data_shape.end()-1, 1, std::multiplies<int>());
    int row_size = data_shape.back();
    
    // Convolve the original transient data with the previous pulses and store it as a complex value
    xt::xarray<std::complex<float>> transient_data = xt::zeros<std::complex<float>>(data_shape);
    #pragma omp parallel for
    for (int row = 0; row < rows; row++)
    {
        for (int t = 0; t < row_size; t++)
        {
            std::complex<float> tmp(0.0f, 0.0f);
            for (int pulse_index = 0; pulse_index < pulse_size; pulse_index++)
            {
                int read_index = t + pulse_index - pulse_size / 2;
                if (read_index > 0 && read_index < row_size)
                {
                    tmp += std::complex<float>(transient_data_in.data()[row*row_size+read_index] * cos_pulse[pulse_index], 
                                               transient_data_in.data()[row*row_size+read_index] * sin_pulse[pulse_index]);
                }
            }
            transient_data.data()[row*row_size+t] = tmp;
        }
    }

    return transient_data;
}


std::tuple<float, float> get_time_limits(const xt::xarray<float>& laser_position,
                                               const xt::xarray<float>& laser_grid_positions,
                                               const xt::xarray<uint32_t>& laser_grid_points,
                                               const xt::xarray<float>& camera_position,
                                               const xt::xarray<float>& camera_grid_positions,
                                               const xt::xarray<uint32_t>& camera_grid_points,
                                               const xt::xarray<float>& volume_size,
                                               const xt::xarray<float>& volume_position,
                                               float t0, uint32_t T, float deltaT,
                                               bool assume_row_major)
{
    const uint32_t num_non_nan_laser_points = 
        xt::sum(!xt::isnan(xt::view(laser_grid_positions, xt::all(), xt::all(), 0)))[0];
    const uint32_t num_non_nan_camera_points = 
        xt::sum(!xt::isnan(xt::view(camera_grid_positions, xt::all(), xt::all(), 0)))[0];

    std::vector<uint32_t> sum_plane;
    if (assume_row_major)
        sum_plane = {0, 1};
    else
        sum_plane = {1, 2};
        
    xt::xarray<float> laser_grid_center = xt::nansum(laser_grid_positions, sum_plane) / num_non_nan_laser_points;
    xt::xarray<float> camera_grid_center = xt::nansum(camera_grid_positions, sum_plane) / num_non_nan_camera_points;

    // Get the minimum and maximum distance traveled to reduce memory footprint
    float min_T_index = 0, max_T_index = 0;
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
        min_T_index = min_distance / deltaT;
        max_T_index = max_distance / deltaT;
    }
    return std::make_tuple(min_T_index, max_T_index);
}

template <typename AT>
xt::xarray<AT> get_transient_chunk(const xt::xarray<AT>& transient_data,
                                   const xt::xarray<uint32_t>& laser_grid_points,
                                   const xt::xarray<uint32_t>& camera_grid_points,
                                   float min_T_index, float max_T_index,
                                   bool is_confocal, bool assume_row_major)
{
    min_T_index = floor(min_T_index);
    max_T_index = ceil(max_T_index);
    auto dshape = transient_data.shape();
    dshape.back() = (size_t) (max_T_index - min_T_index);
    size_t chunk_time_from = min_T_index > 0.0 ? 0 : -min_T_index;
    auto chunk_time_to = xt::all();

    size_t orig_time_from = min_T_index > 0 ? min_T_index : 0;
    size_t orig_time_to = max_T_index;

    // Set the chunk size to match the time range that will be accessed in backprojection
    // This includes expanding the chunk with zeros
    xt::xarray<AT> transient_chunk = xt::zeros<AT>(dshape);

    if (is_confocal)
    {
        if (assume_row_major)
        {
            xt::view(transient_chunk, xt::all(), xt::all(), xt::range(chunk_time_from, chunk_time_to)) = 
                xt::view(transient_data, xt::all(), xt::all(), xt::range(orig_time_from, orig_time_to));
        }
        else
        {
            #pragma omp parallel for collapse(2)
            for (int32_t lx = 0; lx < laser_grid_points[0]; lx++)
                for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
                {
                    xt::view(transient_chunk, lx, ly, xt::all()) = xt::view(transient_data, xt::range(orig_time_from, orig_time_to), ly, lx);
                }
        }
    }
    else
    {
        if (assume_row_major)
        {
            xt::view(transient_chunk, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(chunk_time_from, chunk_time_to)) =
                xt::view(transient_data, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(orig_time_from, orig_time_to));
        }
        else
        {
            #pragma omp parallel for collapse(4)
            for (uint32_t lx = 0; lx < laser_grid_points[0]; lx++)
                for (uint32_t ly = 0; ly < laser_grid_points[1]; ly++)
                    for (uint32_t cx = 0; cx < camera_grid_points[0]; cx++)
                        for (uint32_t cy = 0; cy < camera_grid_points[1]; cy++)
                        {
                            xt::view(transient_chunk, lx, ly, cx, cy, xt::range(chunk_time_from, chunk_time_to)) = xt::view(transient_data, xt::range(orig_time_from, orig_time_to), cy, cx, ly, lx);
                        }
        }
    }
    // transient_chunk = xt::nan_to_num(transient_chunk);
    return std::move(transient_chunk);
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
    const xt::xarray<float> wall_normal=xt::xarray<float>{0,0,0},
    bool use_octree = false)
{
    auto tdata_shape = transient_data.shape();
    uint32_t T = tdata_shape[assume_row_major ? tdata_shape.size() - 1 : 0];

    xt::xarray<uint32_t> camera_grid_points({2});
    xt::xarray<uint32_t> laser_grid_points({2});

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

    // Get the minimum and maximum distance traveled to transfer as little memory to the GPU as possible
    float min_T_index, max_T_index;
    std::tie(min_T_index, max_T_index) = get_time_limits(
        laser_position,
        laser_grid_positions,
        laser_grid_points,
        camera_position,
        camera_grid_positions,
        camera_grid_points,
        volume_size,
        volume_position,
        t0, T, deltaT,
        assume_row_major
    );

    std::vector<ppd> scanned_pairs = precompute_distances(laser_position,
                                                        laser_grid_positions,
                                                        laser_grid_points,
                                                        camera_position,
                                                        camera_grid_positions,
                                                        camera_grid_points,
                                                        is_confocal);


    // Compute the reconstruction volume so that it is aligned to the relay-wall
    xt::xarray<float> x({1,0,0});
    xt::xarray<float> y({0,1,0});
    xt::xarray<float> z({0,0,1});

    if (xt::any(xt::not_equal(wall_normal, 0.0f)))
    {   
        y = -wall_normal;
        x = xt::linalg::cross(y,z);
        z = xt::linalg::cross(y,x);
        x = x / xt::sqrt(xt::sum(x*x));
        y = y / xt::sqrt(xt::sum(y*y));
        z = z / xt::sqrt(xt::sum(z*z));
    }

    xt::xarray<float> volume_zero_pos = volume_position - volume_size / 2;
    xt::xarray<float> voxel_inc = xt::stack(xt::xtuple(
                                    (x * volume_size) / (voxels_per_side - 1), 
                                    (y * volume_size) / (voxels_per_side - 1),
                                    (z * volume_size) / (voxels_per_side - 1)));

    std::cout << "Slicing transient data chunk" << std::flush;

    xt::xarray<float> transient_chunk = get_transient_chunk(
        transient_data, laser_grid_points, camera_grid_points, 
        min_T_index, max_T_index, is_confocal, assume_row_major);

    uint32_t total_transient_size = std::accumulate(transient_chunk.shape().begin(), 
                                                    transient_chunk.shape().end(), 1, 
                                                    std::multiplies<uint32_t>());
    t0 = t0 + min_T_index * deltaT;
    std::cout << " Done!" << std::endl;

    uint32_t chunkedT = (uint32_t)(max_T_index - min_T_index);
    xt::xarray<float> voxel_volume = xt::zeros<float>(voxels_per_side);
    if (use_octree)
    {
        call_cuda_octree_backprojection(transient_chunk.data(),
                                        total_transient_size,
                                        chunkedT,
                                        scanned_pairs,
                                        voxel_volume.data(),
                                        voxels_per_side.data(),
                                        volume_zero_pos.data(),
                                        voxel_inc.data(),
                                        t0, deltaT);
    }
    else
    {
        call_cuda_backprojection(transient_chunk.data(),
                                 total_transient_size,
                                 chunkedT,
                                 scanned_pairs,
                                 voxel_volume.data(),
                                 voxels_per_side.data(),
                                 volume_zero_pos.data(),
                                 voxel_inc.data(),
                                 t0, deltaT);
    }    

    return voxel_volume;
}

xt::xarray<std::complex<float>> phasor_reconstruction(const xt::xarray<float> &transient_data,
                                                      const xt::xarray<float> &camera_grid_positions,
                                                      const xt::xarray<float> &laser_grid_positions,
                                                      const xt::xarray<float> &camera_position,
                                                      const xt::xarray<float> &laser_position,
                                                      float t0, float deltaT, bool is_confocal,
                                                      const xt::xarray<float> &volume_position,
                                                      const xt::xarray<float> volume_size,
                                                      const xt::xarray<uint32_t> voxels_per_side,
                                                      float wavelength=0.04f, bool use_cpu=true,
                                                      bool use_octree = false, bool row_major=true,
                                                      xt::xarray<float> wall_normal={0,0,0})
{
    xt::xarray<float> voxel_size = volume_size / (voxels_per_side - 1);

    xt::xarray<uint32_t> camera_grid_points({2});
    xt::xarray<uint32_t> laser_grid_points({2});
    {
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[0];
        camera_grid_points[1] = t[1];

        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[0];
        laser_grid_points[1] = t[1];
    }

    int T = transient_data.shape().back();
    int min_T_index = 0, max_T_index = 0;
    std::tie(min_T_index, max_T_index) = get_time_limits(
        laser_position, laser_grid_positions, laser_grid_points,
        camera_position, camera_grid_positions, camera_grid_points,
        volume_size, volume_position,
        t0, T, deltaT,
        row_major
    );

        // Compute the reconstruction volume so that it is aligned to the relay-wall
    xt::xarray<float> x({1,0,0});
    xt::xarray<float> y({0,1,0});
    xt::xarray<float> z({0,0,1});

    if (xt::any(xt::not_equal(wall_normal, 0.0f)))
    {   
        y = -wall_normal;
        x = xt::linalg::cross(y,z);
        z = xt::linalg::cross(y,x);
        x = x / xt::sqrt(xt::sum(x*x));
        y = y / xt::sqrt(xt::sum(y*y));
        z = z / xt::sqrt(xt::sum(z*z));
    }

    xt::xarray<float> volume_zero_pos = volume_position - volume_size / 2;
    xt::xarray<float> voxel_inc = xt::stack(xt::xtuple(
                                    (x * volume_size) / (voxels_per_side - 1), 
                                    (y * volume_size) / (voxels_per_side - 1),
                                    (z * volume_size) / (voxels_per_side - 1)));


    xt::xarray<std::complex<float>> complex_transient_data;
    {
        xt::xarray<float> tmp_data = get_transient_chunk(transient_data, 
                                                         laser_grid_points, 
                                                         camera_grid_points, 
                                                         min_T_index, max_T_index,
                                                         is_confocal, row_major);
        complex_transient_data = phasor_pulse(tmp_data, wavelength, deltaT);
    }


    std::vector<ppd> point_pairs = precompute_distances(laser_position,
                                                        laser_grid_positions,
                                                        laser_grid_points,
                                                        camera_position,
                                                        camera_grid_positions,
                                                        camera_grid_points,
                                                        is_confocal);
    uint32_t chunkedT = (uint32_t)(max_T_index - min_T_index);
    t0 = t0 + min_T_index * deltaT;

    ComplexOctreeVolumeF ov(voxels_per_side, volume_size, volume_position);
    if (use_octree)
    {
        if (!use_cpu) std::cerr << "Octree GPU phasor not implemented, defaults to CPU\n";
        octree_backprojection(complex_transient_data, point_pairs, ov, t0, deltaT, chunkedT);
    }
    else
    {
        if (use_cpu)
        {
            classic_backprojection(complex_transient_data, point_pairs, ov, -1, t0, deltaT, chunkedT);
        }
        else
        {
            call_cuda_complex_backprojection(complex_transient_data.data(),
                                             complex_transient_data.size(),
                                             chunkedT, point_pairs,
                                             ov.volume().data(),
                                             voxels_per_side.data(),
                                             volume_zero_pos.data(),
                                             voxel_inc.data(),
                                             t0, deltaT);
        }
    }
    
    return ov.volume();
}


} // namespace bp

#endif
