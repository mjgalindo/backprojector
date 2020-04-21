#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "backproject.hpp"

#include <iostream>
namespace py = pybind11;
using namespace pybind11::literals;

inline xt::pyarray<float> backproject(
    const xt::pyarray<float> &transient_data,
    const xt::pyarray<float> &camera_grid_positions,
    const xt::pyarray<float> &laser_grid_positions,
    const xt::pyarray<float> &camera_position,
    const xt::pyarray<float> &laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    const xt::pyarray<float> &volume_position,
    const xt::pyarray<float> &volume_size,
    const xt::pyarray<uint32_t> &voxels_per_side,
    const xt::pyarray<float> &wall_normal,
    bool assume_row_major = false,
    bool use_cpu = false)
{
    return bp::backproject(transient_data,
                camera_grid_positions,
                laser_grid_positions,
                camera_position,
                laser_position,
                t0, deltaT, 
                is_confocal ? nlos::CaptureStrategy::Confocal : nlos::CaptureStrategy::Exhaustive,
                volume_position,
                volume_size,
                voxels_per_side,
                use_cpu ? nlos::Compute::CPU : nlos::Compute::GPU,
                assume_row_major ? nlos::DataOrder::RowMajor : nlos::DataOrder::ColumnMajor,
                nlos::VolumeAccess::Naive,
                wall_normal);
}

inline xt::pyarray<std::complex<float>> phasor(
    const xt::pyarray<float> &transient_data,
    const xt::pyarray<float> &camera_grid_positions,
    const xt::pyarray<float> &laser_grid_positions,
    const xt::pyarray<float> &camera_position,
    const xt::pyarray<float> &laser_position,
    float t0,
    float deltaT,
    bool is_confocal,
    const xt::pyarray<float> &volume_position,
    const xt::pyarray<float> &volume_size,
    const xt::pyarray<uint32_t> &voxels_per_side,
    float wavelength,
    bool assume_row_major = false,
    bool use_cpu = false)
{
    return bp::phasor_reconstruction(transient_data,
                                     camera_grid_positions,
                                     laser_grid_positions,
                                     camera_position,
                                     laser_position,
                                     t0, deltaT,
                                     is_confocal ? nlos::CaptureStrategy::Confocal : nlos::CaptureStrategy::Exhaustive,
                                     volume_position,
                                     volume_size[0],
                                     voxels_per_side[0],
                                     wavelength,
                                     use_cpu ? nlos::Compute::CPU : nlos::Compute::GPU,
                                     nlos::VolumeAccess::Naive,
                                     assume_row_major ? nlos::DataOrder::RowMajor : nlos::DataOrder::ColumnMajor);
}

// Python Module and Docstrings

PYBIND11_MODULE(nlosbpy, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Python binding for NLOS backprojection and phasor fields reconstructions
    )pbdoc";

    m.def("backproject", backproject, "Backprojects a transient capture to get an unfiltered volume.",
          "transient_data"_a,
          "camera_grid_positions"_a,
          "laser_grid_positions"_a,
          "camera_position"_a,
          "laser_position"_a,
          "t0"_a,
          "deltaT"_a,
          "is_confocal"_a,
          "volume_position"_a,
          "volume_size"_a,
          "voxels_per_side"_a,
          "wall_normal"_a,
          "assume_row_major"_a=false,
          "use_cpu"_a=false);

    m.def("phasor", phasor, "Reconstruct a transient capture using phasor fields.",
          "transient_data"_a,
          "camera_grid_positions"_a,
          "laser_grid_positions"_a,
          "camera_position"_a,
          "laser_position"_a,
          "t0"_a,
          "deltaT"_a,
          "is_confocal"_a,
          "volume_position"_a,
          "volume_size"_a,
          "voxels_per_side"_a,
          "wavelength"_a,
          "assume_row_major"_a=false,
          "use_cpu"_a=false);
}
