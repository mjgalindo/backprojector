#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "backproject.hpp"

#include <iostream>
namespace py = pybind11;


void doshapeprint(const xt::pyarray<float>& thing)
{
    for (auto c: thing.shape())
    std::cout << c << ' ';
    std::cout << std::endl;
}

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
    bool assume_row_major = false,
    bool use_cpu = false)
{
    if (use_cpu)
    {
        // THIS DOES NOT WORK!! And even when it worked it was slow
        return bp::backproject(transient_data,
                    camera_grid_positions,
                    laser_grid_positions,
                    camera_position,
                    laser_position,
                    t0, deltaT, is_confocal,
                    volume_position,
                    volume_size[0],
                    voxels_per_side[0]);
    }
    else
    {
        return bp::gpu_backproject(transient_data,
                    camera_grid_positions,
                    laser_grid_positions,
                    camera_position,
                    laser_position,
                    t0, deltaT, is_confocal,
                    volume_position,
                    volume_size,
                    voxels_per_side,
                    assume_row_major);
    }
}

// Python Module and Docstrings

PYBIND11_MODULE(nlosbpy, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Python binding for NLOS backprojection

        .. currentmodule:: nlosbpy

        .. autosummary::
           :toctree: _generate
            backproject
    )pbdoc";

    m.def("backproject", backproject, "Backprojects a transient capture to get an unfiltered volume.");
}