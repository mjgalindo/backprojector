#ifndef BACKPROJECT_HPP
#define BACKPROJECT_HPP

#define XTENSOR_ENABLE_XSIMD

#include <complex>
#include <math.h>
#include <limits>
#include <xtensor/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
#include <omp.h>

#include "backproject_cuda.hpp"
#include "octree_volume.hpp"
#include "nlos_enums.hpp"

#if __linux__
#include "tqdm.h"
#endif

namespace bp
{
using namespace nlos;

/**
 * @brief Distance between two 3D points
 * 
 * Computes the euclidean distance between two points in 3D space.
 * 
 * @param p1 first 3D point
 * @param p2 second 3D point
 * 
 * @tparam V1 The type of a container holding 3 elements.
 * @tparam V2 The type of a container holding 3 elements.
 * @return distance between the points
 */
template <typename V1, typename V2>
double distance(const V1 &p1, const V2 &p2)
{
    std::array<double, 3> tmp;
    for (int i = 0; i < 3; i++)
    {
        tmp[i] = p1[i] - p2[i];
        tmp[i] = tmp[i] * tmp[i];
    }

    return sqrt(tmp[0] + tmp[1] + tmp[2]);
}

/**
 * @brief Sums all radiance matching a single voxel
 * 
 * Sums the radiance in a transient image for all camera and laser
 * pairs. Considers the transient data to have point_pairs.size() 
 * rows of size T, and accesses the value that is at a distance
 * given by the pairs and the voxel position. It also considers the
 * values in a diameter given by access_width.
 * 
 * @param transient_data row-major transient image with last dimension length T
 * @param point_pairs camera / laser position pairs and precomputed distances. point_pairs.size() must be <= T
 * @param voxel_position 3D position of the target voxel
 * @param t0 first instant recorded in transient_data in meters
 * @param deltaT exposure for each pixel in meters
 * @param T length of each transient_data row
 * @param access_width diameter to integrate over in each transient_data row in meters
 * 
 * @tparam AT Type of the values in the transient_data. Can be real (float, double) or complex numbers (std::complex)
 * @return integrated radiance over all captured points
 */
template <typename AT>
AT accumulate_radiance(const xt::xarray<AT>& transient_data,
                       const std::vector<CameraLaserPair>& point_pairs,
                       const xt::xarray<float>& voxel_position,
                       float t0, float deltaT, uint32_t T, float access_width)
{
    AT radiance_sum = AT{};
    for (uint32_t pairId = 0; pairId < point_pairs.size(); pairId++)
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

/**
 * @brief Checks if the value is within the threshold
 * 
 * Returns true when the value is over the threshold.
 * 
 * @param value value to check
 * @param threshold minimum valid value 
 * 
 * @tparam AT Type for the values. Can be real (float, double) or complex numbers (std::complex)
 * @return true if value >= threshold, false otherwise
 */
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

/**
 * @brief Backproject transient data into a specified volume.
 * 
 * Runs the backprojection algorithm over the given volume (adding to it).
 * 
 * @param transient_data row-major transient image with last dimension length T
 * @param point_pairs camera / laser position pairs and precomputed distances. point_pairs.size() must be <= 
 * @param volume In/out 3D texture. It's voxels will accumulate the radiance over the whole transient image.
 * @param depth volume lookup depth
 * @param t0 first time instant recorded in transient_data
 * @param deltaT time exposure for each pixel in seconds
 * @param T length of each transient_data row
 * @param threshold minimum value to consider running backprojection in a voxel
 * @param verbose flag to enable/disable progress information
 * 
 * @tparam AT Type for the values. Can be real (float, double) or complex numbers (std::complex)
 */
template <typename AT>
void classic_backprojection(const xt::xarray<AT>& transient_data, 
                            const std::vector<CameraLaserPair>& point_pairs,
                            OctreeVolumeF<AT>& volume, int depth,
                            float t0, float deltaT, uint32_t T,
                            AT threshold = 0.0f, bool verbose=true)
{
    auto max_voxels = volume.max_voxels(depth);
    xt::xarray<float> voxel_size = volume.voxel_size_at(depth);
    float voxel_diagonal = xt::sqrt(xt::sum(voxel_size * voxel_size))[0];

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
        Iter3D iter(max_voxels);
        int total_length = iter.total_length();
        float thread_iterations = total_length / (float) nthreads;
        uint32_t from = std::floor(threadId * thread_iterations);
        // If the length is not divisible by the number of threads, 
        // the last thread gets less work
        uint32_t to = from + std::ceil(thread_iterations);
        if (threadId == nthreads - 1) to = iter.total_length();
        int local_avoided = 0;
        iter.jump_to(from);
        for (uint32_t id = from; id < to; ++id)
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

/**
 * @brief Backproject transient data into a specified volume.
 * 
 * Runs the backprojection algorithm over the given volume (adding to it).
 * 
 * @param transient_data row-major transient image with last dimension length T
 * @param point_pairs camera / laser position pairs and precomputed distances. point_pairs.size() must be <= 
 * @param volume_size size of the volume to backproject in XYZ units in space.
 * @param volume_position XYZ center position of the volume to backproject
 * @param volume In/out 3D texture. It's voxels will accumulate the radiance over the whole transient image.
 * @param t0 first time instant recorded in transient_data
 * @param deltaT time exposure for each pixel in seconds
 * @param T length of each transient_data row
 * @param threshold minimum value to consider running backprojection in a voxel
 * @param verbose flag to enable/disable progress information
 * 
 * @tparam AT Type for the values. Can be real (float, double) or complex numbers (std::complex)
 */
template <typename AT>
void classic_backprojection(const xt::xarray<AT>& transient_data, 
                            const std::vector<CameraLaserPair>& point_pairs,
                            const xt::xarray<float>& volume_size,
                            const xt::xarray<float>& volume_position,
                            xt::xarray<AT>& volume,
                            float t0, float deltaT, uint32_t T, bool verbose=true)
{
    OctreeVolumeF<AT> ov(volume, volume_size, volume_position);
    classic_backprojection(transient_data, point_pairs, ov, -1, t0, deltaT, T, 0.0f, verbose);
    volume = ov.volume();
}

/**
 * @brief Backproject transient data into a specified volume using an Octree hierarchy.
 * 
 * Backproject transient data into a specified volume using an Octree hierarchy for performance.
 * 
 * @param transient_data row-major transient image with last dimension length T
 * @param point_pairs camera / laser position pairs and precomputed distances. point_pairs.size() must be <= 
 * @param volume In/out 3D texture. It's voxels will accumulate the radiance over the whole transient image.
 * @param t0 first time instant recorded in transient_data
 * @param deltaT time exposure for each pixel in seconds
 * @param T length of each transient_data row
 * @param verbose flag to enable/disable progress information
 * 
 * @tparam AT Type for the values. Can be real (float, double) or complex numbers (std::complex)
 */
template <typename AT>
void octree_backprojection(const xt::xarray<AT>& transient_data, 
                           const std::vector<CameraLaserPair>& point_pairs,
                           OctreeVolumeF<AT>& volume,
                           float t0, float deltaT, uint32_t T, bool verbose=true)
{
    uint32_t depth = 3;
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

/**
 * @brief Backproject transient data into a specified volume using an Octree hierarchy.
 * 
 * Backproject transient data into a specified volume using an Octree hierarchy for performance.
 * 
 * @param transient_data row-major transient image with last dimension length T
 * @param point_pairs camera / laser position pairs and precomputed distances. point_pairs.size() must be <= 
 * @param volume_size size of the volume to backproject in XYZ units in space.
 * @param volume_position XYZ center position of the volume to backproject
 * @param volume In/out 3D texture. It's voxels will accumulate the radiance over the whole transient image.
 * @param t0 first time instant recorded in transient_data
 * @param deltaT time exposure for each pixel in seconds
 * @param T length of each transient_data row
 * @param verbose flag to enable/disable progress information
 * 
 * @tparam AT Type for the values. Can be real (float, double) or complex numbers (std::complex)
 */
template <typename AT>
void octree_backprojection(const xt::xarray<AT>& transient_data, 
                            const std::vector<CameraLaserPair>& point_pairs,
                            const xt::xarray<float>& volume_size,
                            const xt::xarray<float>& volume_position,
                            xt::xarray<AT>& volume,
                            float t0, float deltaT, uint32_t T, bool verbose=true)
{
    OctreeVolumeF<AT> ov(volume, volume_size, volume_position);
    octree_backprojection(transient_data, point_pairs, ov, t0, deltaT, T, verbose);
    volume = ov.volume();
}


/**
 * @brief Precomputes Camera-wall and laser wall distsances
 * 
 * Computes the distance between the camera and all camera points and the laser
 * and all laser points. The result is stored by pairs, with all camera points
 * being computed for all laser points (raw-major order with [laser_x, laser_y, cam_x, cam_y])
 * 
 * @param laser_position XYZ position of the laser
 * @param laser_grid_positions XYZ positions of all the lasers. Size -> [lx, ly, 3]
 * @param laser_grid_points XY dimensions of the laser grid ([lx, ly]). Size -> 2
 * @param camera_position XYZ position of the camera
 * @param camera_grid_positions XYZ positions of all the cameras. Size -> [lx, ly, 3]
 * @param camera_grid_points XY dimensions of the camera grid ([lx, ly]). Size -> 2
 * @param capture strategy used to capture the points. In a confocal setup, laser and 
 * camera grid positions are equal
 * @return vector of all the combinations of camera and laser points (for the capture strategy)
 * and the distance form the wall to them.
 */ 
std::vector<CameraLaserPair> precompute_distances(const xt::xarray<float>& laser_position,
                                                  const xt::xarray<float>& laser_grid_positions,
                                                  const xt::xarray<uint32_t>& laser_grid_points,
                                                  const xt::xarray<float>& camera_position,
                                                  const xt::xarray<float>& camera_grid_positions,
                                                  const xt::xarray<uint32_t>& camera_grid_points,
                                                  CaptureStrategy capture)
{
    uint32_t number_of_pairs = capture == CaptureStrategy::Confocal ? 
        camera_grid_points[0] * camera_grid_points[1] : 
        camera_grid_points[0] * camera_grid_points[1] * 
        laser_grid_points[0] * laser_grid_points[1];

    std::vector<CameraLaserPair> point_pairs(number_of_pairs);

    // Calculate distances
    switch(capture)
    {
        case Exhaustive:
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
                            xt::xarray<float> camera_point = xt::view(camera_grid_positions, cx, cy, xt::all());
                            uint32_t p_id = lx * laser_grid_points[1] * camera_grid_points[0] * camera_grid_points[1] +
                                            ly * camera_grid_points[0] * camera_grid_points[1] +
                                            cx * camera_grid_points[1] +
                                            cy;
                            std::copy(laser_point.begin(), laser_point.end(), point_pairs[p_id].laser_point);
                            std::copy(camera_point.begin(), camera_point.end(), point_pairs[p_id].camera_point);
                            point_pairs[p_id].camera_wall = distance(camera_position, camera_point);
                            point_pairs[p_id].laser_wall = distance(laser_position, laser_point);
                        }
                    }
                }
            }
            break;
        case Confocal:
            #pragma omp parallel for collapse(2)
            for (size_t lx = 0; lx < laser_grid_points[0]; lx++)
            {
                for (size_t ly = 0; ly < laser_grid_points[1]; ly++)
                {
                    auto camera_point = xt::view(camera_grid_positions, lx, ly, xt::all());
                    uint32_t p_id = lx * laser_grid_points[1] + ly;
                    std::copy(camera_point.begin(), camera_point.end(), point_pairs[p_id].laser_point);
                    std::copy(camera_point.begin(), camera_point.end(), point_pairs[p_id].camera_point);
                    float laser_camera_wall = distance(camera_position, camera_point);
                    point_pairs[p_id].camera_wall = laser_camera_wall;
                    point_pairs[p_id].laser_wall = laser_camera_wall;
                }
            }
            break;
    }
        
    return std::move(point_pairs);
}

/**
 * @brief Creates the complex phasor field signal from an input transient volume.
 * 
 * 
 * @param transient_data_in transient dataset that will be convolved to have a real and complex part
 * @param wavelength
 * @param deltaT
 * @param times
 * @return Complex phasor field data
 */ 
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


/**
 * @brief Get the corner object
 * 
 * @tparam T 
 * @param volume 
 * @param mask 
 * @return xt::xarray<T> 
 */
template <typename T>
xt::xarray<T> get_corner(xt::xarray<T> volume, uint32_t mask)
{
    const auto shape = volume.shape();
    const int ndim = shape.size()-1;

    xt::xstrided_slice_vector sv(ndim);
    for (uint32_t i = 0; i < sv.size(); i++)
    {
        sv[i] = (mask & 1 << (ndim - 1 - i)) ? shape[i]-1 : 0;
    }
    sv.push_back(xt::all());
    return xt::strided_view(volume, sv);
}

/**
 * @brief Get the corner object
 * 
 * @tparam T 
 * @param v_position 
 * @param v_size 
 * @param mask 
 * @return xt::xarray<T> 
 */
template <typename T>
xt::xarray<T> get_corner(xt::xarray<T> v_position, xt::xarray<T> v_size, uint32_t mask)
{
    const int ndim = v_size.size();
    xt::xarray<T> tmp_multiplier = xt::zeros<T>({(size_t) ndim});
    for (int i = 0; i < ndim; i++)
    {
        tmp_multiplier[i] = (mask & 1 << (ndim - 1 - i)) ? 1.0f : 0.0f;
    }
    auto lower_corner = v_position - v_size / 2;
    return (v_position - v_size / 2) + tmp_multiplier * v_size;
}

/**
 * @brief Get the nd corners object
 * 
 * @tparam T 
 * @param ndim_volume 
 * @return xt::xarray<T> 
 */
template <typename T>
xt::xarray<T> get_nd_corners(xt::xarray<T> ndim_volume)
{
    const auto shape = ndim_volume.shape();
    const int ndim = shape.size()-1;
    const int num_corners = 1 << ndim;
    xt::xarray<T> corners = xt::empty<T>({(size_t) num_corners, (size_t) shape.back()});
    for (int i = 0; i < num_corners; i++)
    {
        xt::view(corners, i, xt::all()) = get_corner(ndim_volume, i);
    }

    return corners;
}

/**
 * @brief Get the nd corners object
 * 
 * @tparam T 
 * @param v_position 
 * @param v_size 
 * @return xt::xarray<T> 
 */
template <typename T>
xt::xarray<T> get_nd_corners(xt::xarray<T> v_position, xt::xarray<T> v_size)
{
    const int ndim = v_size.size();
    const int num_corners = 1 << ndim;
    xt::xarray<T> corners = xt::empty<T>({(size_t) num_corners, (size_t) ndim});
    for (int i = 0; i < num_corners; i++)
    {
        xt::view(corners, i, xt::all()) = get_corner(v_position, v_size, i);
    }

    return corners;
}

/**
 * @brief Get the min max distances object
 * 
 * @tparam T 
 * @param set_a 
 * @param set_b 
 * @return std::tuple<T, T> 
 */
template <typename T>
std::tuple<T, T> get_min_max_distances(xt::xarray<T> set_a, xt::xarray<T> set_b)
{
    const auto shape_a = set_a.shape();
    const auto shape_b = set_b.shape();

    float min_dist = std::numeric_limits<float>::max();
    float max_dist = 0.0f;

    for (uint32_t i = 0; i < shape_a[0]; i++)
    {
        for (uint32_t j = 0; j < shape_b[0]; j++)
        {
            float dist = distance(xt::view(set_a, i, xt::all()), xt::view(set_b, j, xt::all()));
            min_dist = std::min({dist, min_dist});
            max_dist = std::max({dist, max_dist});
        }
    }
    return std::make_tuple(min_dist, max_dist);
}

/**
 * @brief Get the time limits object
 * 
 * @param laser_position 
 * @param laser_grid_positions 
 * @param laser_grid_points 
 * @param camera_position 
 * @param camera_grid_positions 
 * @param camera_grid_points 
 * @param volume_size 
 * @param volume_position 
 * @param t0 
 * @param T 
 * @param deltaT 
 * @param data_order 
 * @return std::tuple<float, float> 
 */
std::tuple<float, float> get_time_limits(const xt::xarray<float>& laser_position,
                                         const xt::xarray<float>& laser_grid_positions,
                                         const xt::xarray<uint32_t>& laser_grid_points,
                                         const xt::xarray<float>& camera_position,
                                         const xt::xarray<float>& camera_grid_positions,
                                         const xt::xarray<uint32_t>& camera_grid_points,
                                         const xt::xarray<float>& volume_size,
                                         const xt::xarray<float>& volume_position,
                                         float t0, uint32_t T, float deltaT,
                                         DataOrder data_order)
{
    const uint32_t num_non_nan_laser_points = 
        xt::sum(!xt::isnan(xt::view(laser_grid_positions, xt::all(), xt::all(), 0)))[0];
    const uint32_t num_non_nan_camera_points = 
        xt::sum(!xt::isnan(xt::view(camera_grid_positions, xt::all(), xt::all(), 0)))[0];

    std::vector<uint32_t> sum_plane;

    switch (data_order)
    {
        case RowMajor:
            sum_plane = {0, 1};
            break;
        case ColumnMajor:
            sum_plane = {1, 2};
            break;
    }
        
    xt::xarray<float> laser_grid_center = xt::nansum(laser_grid_positions, sum_plane) / num_non_nan_laser_points;
    xt::xarray<float> camera_grid_center = xt::nansum(camera_grid_positions, sum_plane) / num_non_nan_camera_points;

    // Get the minimum and maximum distance traveled to reduce memory footprint
    float min_T_index = 0, max_T_index = 0;
    {
        auto lgp_view = data_order == RowMajor ? laser_grid_positions : xt::transpose(laser_grid_positions);
        auto cgp_view = data_order == ColumnMajor ? camera_grid_positions : xt::transpose(camera_grid_positions);

        xt::xarray<float> las_corners = get_nd_corners(lgp_view);
        xt::xarray<float> cam_corners = get_nd_corners(cgp_view);
        xt::xarray<float> vol_corners = get_nd_corners(volume_position, volume_size);

        float min_las_wall, max_las_wall, min_cam_wall, max_cam_wall, min_lgrid_vol, max_lgrid_vol, min_cgrid_vol, max_cgrid_vol;

        std::tie(min_las_wall, max_las_wall) = get_min_max_distances(laser_position, las_corners);
        std::tie(min_cam_wall, max_cam_wall) = get_min_max_distances(camera_position, cam_corners);
        std::tie(min_lgrid_vol, max_lgrid_vol) = get_min_max_distances(vol_corners, las_corners);
        std::tie(min_cgrid_vol, max_cgrid_vol) = get_min_max_distances(vol_corners, cam_corners);

        float min_distance = min_las_wall + min_cam_wall + min_lgrid_vol + min_cgrid_vol;
        float max_distance = max_las_wall + max_cam_wall + max_lgrid_vol + max_cgrid_vol;
        // Adjust distances by the minimum recorded distance.
        min_distance -= t0;
        max_distance -= t0;
        min_T_index = min_distance / deltaT;
        max_T_index = max_distance / deltaT;
    }
    return std::make_tuple(min_T_index-100, max_T_index+100);
}

/**
 * @brief Get the transient chunk object
 * 
 * @tparam AT 
 * @param transient_data 
 * @param laser_grid_points 
 * @param camera_grid_points 
 * @param min_T_index 
 * @param max_T_index 
 * @param capture 
 * @param data_order 
 * @return xt::xarray<AT> 
 */
template <typename AT>
xt::xarray<AT> get_transient_chunk(const xt::xarray<AT>& transient_data,
                                   const xt::xarray<uint32_t>& laser_grid_points,
                                   const xt::xarray<uint32_t>& camera_grid_points,
                                   float min_T_index, float max_T_index,
                                   CaptureStrategy capture, DataOrder data_order)
{
    min_T_index = floor(min_T_index);
    max_T_index = ceil(max_T_index);
    auto dshape = transient_data.shape();
    size_t orig_T = transient_data.shape().back();
    dshape.back() = (size_t) (max_T_index - min_T_index);

    size_t orig_time_from = min_T_index > 0.0 ? min_T_index : 0;
    size_t orig_time_to = std::min({orig_T, (size_t) max_T_index});

    size_t chunk_time_from = min_T_index > 0.0 ? 0 : abs(min_T_index);
    size_t chunk_time_to = orig_time_to - orig_time_from + chunk_time_from;

    assert(chunk_time_to - chunk_time_from == orig_time_to - orig_time_from);
    // Set the chunk size to match the time range that will be accessed in backprojection
    // This includes expanding the chunk with zeros
    xt::xarray<AT> transient_chunk = xt::zeros<AT>(dshape);

    if (capture == Confocal)
    {
        if (data_order == RowMajor)
        {
            xt::view(transient_chunk, xt::all(), xt::all(), xt::range(chunk_time_from, chunk_time_to)) = 
                xt::view(transient_data, xt::all(), xt::all(), xt::range(orig_time_from, orig_time_to));
        }
        else if (data_order == ColumnMajor)
        {
            xt::view(transient_chunk, xt::all(), xt::all(), xt::range(chunk_time_from, chunk_time_to)) = 
                xt::view(xt::transpose(transient_data), xt::all(), xt::all(), xt::range(orig_time_from, orig_time_to));
        }
    }
    else if (capture == Exhaustive)
    {
        if (data_order == RowMajor)
        {
            xt::view(transient_chunk, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(chunk_time_from, chunk_time_to)) =
                xt::view(transient_data, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(orig_time_from, orig_time_to));
        }
        else if (data_order == ColumnMajor)
        {
            xt::view(transient_chunk, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(chunk_time_from, chunk_time_to)) = 
                xt::view(xt::transpose(transient_data), xt::all(), xt::all(), xt::all(), xt::all(), xt::range(orig_time_from, orig_time_to));
        }
    }
    return std::move(transient_chunk);
}

/**
 * @brief 
 * 
 * @param transient_data 
 * @param camera_grid_positions 
 * @param laser_grid_positions 
 * @param camera_position 
 * @param laser_position 
 * @param t0 
 * @param deltaT 
 * @param capture 
 * @param volume_position 
 * @param volume_size 
 * @param voxels_per_side 
 * @param compute 
 * @param data_order 
 * @param vol_access 
 * @param wall_normal 
 * @return xt::xarray<float> 
 */
xt::xarray<float> backproject(
    const xt::xarray<float> &transient_data,
    const xt::xarray<float> &camera_grid_positions,
    const xt::xarray<float> &laser_grid_positions,
    const xt::xarray<float> &camera_position,
    const xt::xarray<float> &laser_position,
    float t0,
    float deltaT,
    CaptureStrategy capture,
    const xt::xarray<float> &volume_position,
    const xt::xarray<float> &volume_size,
    xt::xarray<uint32_t> voxels_per_side,
    Compute compute = Compute::GPU,
    DataOrder data_order = DataOrder::RowMajor,
    VolumeAccess vol_access = VolumeAccess::Naive,
    const xt::xarray<float> wall_normal=xt::xarray<float>{0,0,0})
{
    auto tdata_shape = transient_data.shape();
    uint32_t T = tdata_shape[data_order == RowMajor ? tdata_shape.size() - 1 : 0];

    xt::xarray<uint32_t> camera_grid_points = xt::zeros<uint32_t>({2});
    xt::xarray<uint32_t> laser_grid_points = xt::zeros<uint32_t>({2});

    {
        // camera_grid_positions.shape() is (points_x, points_y, 3)
        auto t = camera_grid_positions.shape();
        camera_grid_points[0] = t[data_order == RowMajor ? 0 : 2];
        camera_grid_points[1] = t[data_order == RowMajor ? 1 : 1];

        // laser_grid_positions.shape() is (points_x, points_y, 3)
        t = laser_grid_positions.shape();
        laser_grid_points[0] = t[data_order == RowMajor ? 0 : 2];
        laser_grid_points[1] = t[data_order == RowMajor ? 1 : 1];
    }

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
        data_order
    );

    std::vector<CameraLaserPair> point_pairs = precompute_distances(laser_position,
                                                        laser_grid_positions,
                                                        laser_grid_points,
                                                        camera_position,
                                                        camera_grid_positions,
                                                        camera_grid_points,
                                                        capture);

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
        min_T_index, max_T_index, capture, data_order);

    float chunked_t0 = t0 + min_T_index * deltaT;
    std::cout << " Done!" << std::endl;

    xt::xarray<float> voxel_volume = xt::zeros<float>(voxels_per_side);

    switch (compute)
    {
        case CPU:
            switch (vol_access)
            {
                case Naive:
                    classic_backprojection(transient_chunk,
                                           point_pairs,
                                           volume_size,
                                           volume_position,
                                           voxel_volume,
                                           chunked_t0,
                                           deltaT,
                                           transient_chunk.shape().back());
                    break;
                case Octree:
                    octree_backprojection(transient_chunk,
                                          point_pairs,
                                          volume_size,
                                          volume_position,
                                          voxel_volume,
                                          chunked_t0,
                                          deltaT,
                                          transient_chunk.shape().back());
                    break;
            }
            break;
        case GPU:
            switch (vol_access)
            {
                case Naive:
                    call_cuda_backprojection(transient_chunk.data(),
                                            transient_chunk.size(),
                                            transient_chunk.shape().back(),
                                            point_pairs,
                                            voxel_volume.data(),
                                            voxels_per_side.data(),
                                            volume_zero_pos.data(),
                                            voxel_inc.data(),
                                            chunked_t0, deltaT);
                    break;
                case Octree:
                    call_cuda_octree_backprojection(transient_chunk.data(),
                                                    transient_chunk.size(),
                                                    transient_chunk.shape().back(),
                                                    point_pairs,
                                                    voxel_volume.data(),
                                                    voxels_per_side.data(),
                                                    volume_zero_pos.data(),
                                                    voxel_inc.data(),
                                                    chunked_t0, deltaT);
                    break;
            }
            break;
    }

    return voxel_volume;
}

/**
 * @brief 
 * 
 * @param transient_data 
 * @param camera_grid_positions 
 * @param laser_grid_positions 
 * @param camera_position 
 * @param laser_position 
 * @param t0 
 * @param deltaT 
 * @param capture 
 * @param volume_position 
 * @param volume_size 
 * @param voxels_per_side 
 * @param wavelength 
 * @param compute 
 * @param vol_access 
 * @param data_order 
 * @param wall_normal 
 * @return xt::xarray<std::complex<float>> 
 */
xt::xarray<std::complex<float>> phasor_reconstruction(const xt::xarray<float> &transient_data,
                                                      const xt::xarray<float> &camera_grid_positions,
                                                      const xt::xarray<float> &laser_grid_positions,
                                                      const xt::xarray<float> &camera_position,
                                                      const xt::xarray<float> &laser_position,
                                                      float t0, float deltaT, 
                                                      CaptureStrategy capture,
                                                      const xt::xarray<float> &volume_position,
                                                      const xt::xarray<float> volume_size,
                                                      const xt::xarray<uint32_t> voxels_per_side,
                                                      float wavelength=0.04f, Compute compute=GPU,
                                                      VolumeAccess vol_access = Naive, 
                                                      DataOrder data_order=RowMajor,
                                                      xt::xarray<float> wall_normal={0,0,0})
{
    xt::xarray<float> voxel_size = volume_size / (voxels_per_side - 1);

    xt::xarray<uint32_t> camera_grid_points = xt::zeros<uint32_t>({2});
    xt::xarray<uint32_t> laser_grid_points = xt::zeros<uint32_t>({2});
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
        data_order
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
                                                         capture, data_order);
        complex_transient_data = phasor_pulse(tmp_data, wavelength, deltaT);
    }

    std::vector<CameraLaserPair> point_pairs = precompute_distances(laser_position,
                                                        laser_grid_positions,
                                                        laser_grid_points,
                                                        camera_position,
                                                        camera_grid_positions,
                                                        camera_grid_points,
                                                        capture);

    float chunked_t0 = t0 + min_T_index * deltaT;

    ComplexOctreeVolumeF ov(voxels_per_side, volume_size, volume_position);
    if (vol_access == Octree)
    {
        if (compute != nlos::Compute::GPU) std::cerr << "Octree GPU phasor not implemented, defaults to CPU\n";
        octree_backprojection(complex_transient_data, point_pairs, ov, t0, deltaT, complex_transient_data.shape().back());
    }
    else
    {
        if (compute == nlos::Compute::CPU)
        {
            classic_backprojection(complex_transient_data, point_pairs, ov, -1, chunked_t0, deltaT, complex_transient_data.shape().back());
        }
        else if (compute == nlos::Compute::GPU)
        {
            call_cuda_complex_backprojection(complex_transient_data.data(),
                                             complex_transient_data.size(),
                                             complex_transient_data.shape().back(), point_pairs,
                                             ov.volume().data(),
                                             voxels_per_side.data(),
                                             volume_zero_pos.data(),
                                             voxel_inc.data(),
                                             chunked_t0, deltaT);
        }
    }
    
    return ov.volume();
}


} // namespace bp

#endif
