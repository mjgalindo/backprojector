#ifndef BACKPROJECT_CUDA
#define BACKPROJECT_CUDA

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xview.hpp>

namespace bp_cuda
{
using vector3 = xt::xtensor_fixed<float, xt::xshape<3>>;

xt::xarray<float> backproject(
    xt::xarray<float> &transient_data,
    const xt::xtensor<float, 3> &camera_grid_positions,
    const xt::xtensor<float, 3> &laser_grid_positions,
    vector3 camera_position,
    vector3 laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    vector3 volume_position,
    float volume_size,
    uint32_t voxels_per_side);
} // namespace bp_cuda
#endif