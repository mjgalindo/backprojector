#include "backproject.hpp"
#include "backproject_cuda.hpp"
#include "nlos_loader.hpp"

/// Example loading and showing some values from an NLOS dataset.
int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage:\n\tbackprojector dataset_file [-b highest_bounce, -res volume resolution, -cpu]\n";
        return 1;
    }
    int highest_bounce = 3;
    int voxel_resolution = 32;
    bool use_cpu = false;
    std::string filename(argv[1]);
    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
        {
            highest_bounce = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-res") == 0 && i + 1 < argc)
        {
            voxel_resolution = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-cpu") == 0)
        {
            use_cpu = true;
        }
    }

    std::vector<uint32_t> bounces(0, highest_bounce - 2);
    for (int i = 3; i <= highest_bounce; i++)
        bounces.push_back(i);

    NLOSData data(filename, bounces);

    bool is_confocal = data.is_confocal[0];
    float deltaT = data.deltat[0];
    float t0 = data.t0[0];
    // transient_data.shape() -> (channels, T, cy, cx, ly, lx)
    xt::xarray<float> transient_data = xt::view(data.data, 0, xt::all());
    // transient_data.shape() -> (T, cy, cx, ly, lx)

    xt::xarray<float> volume;
    if (use_cpu)
    {
        volume = bp::backproject(transient_data,
                                 data.camera_grid_positions,
                                 data.laser_grid_positions,
                                 data.camera_position,
                                 data.laser_position,
                                 t0, deltaT, is_confocal,
                                 data.hidden_volume_position,
                                 data.hidden_volume_size[0] * 2, voxel_resolution);
    }
    else
    {
        volume = bp_cuda::backproject(transient_data,
                                      data.camera_grid_positions,
                                      data.laser_grid_positions,
                                      data.camera_position,
                                      data.laser_position,
                                      t0, deltaT, is_confocal,
                                      data.hidden_volume_position,
                                      data.hidden_volume_size[0] * 2, voxel_resolution);
    }
    char outfile[256] = {};
    sprintf(outfile, "test_%d.hdf5", voxel_resolution);
    xt::dump_hdf5(outfile, "/voxel_volume", volume, xt::file_mode::overwrite);
    std::cout << "Backprojected to " << std::string(outfile) << std::endl;
    return 0;
}
