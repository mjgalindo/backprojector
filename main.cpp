#include "backproject.hpp"
#include "nlos_loader.hpp"

using namespace bp;

/// Example loading and showing some values from an NLOS dataset.
int main(int argc, const char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage:\n\tbackprojector dataset_file [-b highest_bounce]\n";
        return 1;
    }
    int highest_bounce = 3;
    int voxel_resolution = 32;
    std::string filename(argv[1]);
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0 && i+1 < argc) {
            highest_bounce = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "-v") == 0 && i+1 < argc) {
            voxel_resolution = atoi(argv[i+1]);
        }
    }

    std::vector<uint> bounces(0, highest_bounce - 2);
    for (int i = 3; i <= highest_bounce; i++) bounces.push_back(i);
    std::cout << "Loading up to the " << highest_bounce << " bounce" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    NLOSData data(filename, bounces);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Complete scene was read in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    for (const auto d : data.data.shape()){
        std::cout << d << ' ';
    }
    std::cout << std::endl;
    bool is_confocal = data.is_confocal[0];
    float deltaT = data.deltat[0];
    float t0 = data.t0[0];
    xt::xarray<float> transient_data = xt::view(data.data, 0, xt::all());
    auto volume = backproject(transient_data,
                              data.camera_grid_positions,
                              data.laser_grid_positions,
                              data.camera_position,
                              data.laser_position,
                              t0, deltaT, is_confocal,
                              data.hidden_volume_position,
                              data.hidden_volume_size[0]*2, voxel_resolution);
    char outfile[256] = {};
    sprintf(outfile, "test_%d.hdf5", voxel_resolution);
    xt::dump_hdf5(outfile, "/voxel_volume", volume, xt::file_mode::overwrite);
}
