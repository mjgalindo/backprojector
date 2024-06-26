#include "backproject.hpp"
#include "backproject_cuda.hpp"
#include "dataset_loader.hpp"
#include "nlos_dataset.hpp"
#include "args.hxx"
#include <fstream>

using namespace nlos;

template <typename T, unsigned int N>
struct ArrayReader
{
    void operator()(const std::string &name, const std::string &value, xt::xarray<T> &arr)
    {
        uint32_t position = 0;
        arr = xt::empty<float>({N});
        for (uint32_t i = 0; i < N; i++)
        {
            size_t commapos = 0;
            arr[i] = (T)std::stod(value.substr(position), &commapos);
            position += commapos + 1;
            if (position > value.length()) position = 0;
        }
    }
};

int main(int argc, const char *argv[])
{
    int highest_bounce = -1;
    std::string filename(argv[1]);
    xt::xarray<uint32_t> voxel_resolution = {-1u, -1u, -1u};
    xt::xarray<float> volume_position = {NAN, NAN, NAN};
    xt::xarray<float> volume_size = {NAN, NAN, NAN};
    xt::xarray<float> volume_direction = {NAN, NAN, NAN};
    std::string outfile = "";
    bool use_cpu = false;
    bool use_octree = false;
    bool use_phasor_fields = false;
    float wavelength = 0.0f;

    args::ArgumentParser parser("NLOS dataset backprojector.", "Takes a NLOS dataset and returns an unfiltered backprojection volume.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::Positional<std::string> vf_filename(parser, "filename", "The input file name.");
    args::ValueFlag<uint32_t> vf_highest_bounce(parser,
                                                "highest_bounce",
                                                "Maximum bounces to consider in the reconstruction. "
                                                "Valid values start counting from 3 and are bound to the maximum in the dataset (usually 8, defaults to 3).",
                                                {'b', "max-bounce"}, 3);
    args::ValueFlag<xt::xarray<int>, ArrayReader<int, 3u>>
        vf_voxel_resolution(parser,
                            "voxel_resolution",
                            "Resolution in the XYZ axis of the volume.",
                            {'r', "voxel-resolution"}, {32, 32, 32});
    args::ValueFlag<xt::xarray<float>, ArrayReader<float, 3u>>
        vf_volume_position(parser,
                           "volume_position",
                           "Center of the hidden volume. Defaults to the position read from the dataset.",
                           {'c', "volume-position"}, {NAN, NAN, NAN});
    args::ValueFlag<xt::xarray<float>, ArrayReader<float, 3u>>
        vf_volume_size(parser,
                       "volume_size",
                       "Axis-aligned size of the volume in XYZ. Defaults to the size read from the dataset.",
                       {'s', "volume-size"}, {NAN, NAN, NAN});
    args::ValueFlag<xt::xarray<float>, ArrayReader<float, 3u>>
        vf_volume_direction(parser,
                            "volume_direction",
                            "NOT IMPLEMENTED!! Direction vector in which the volume will be \"grown\". Defaults to keeping the volume axis aligned.",
                            {'d', "volume-direction"}, {1, 0, 0});
    args::ValueFlag<std::string> vf_outfile(parser,
                                            "outfile",
                                            "Output file for the result.",
                                            {'o', "output"}, "");

    args::Flag vf_use_cpu(parser, "use_cpu", "Flag to force CPU backprojection", {"cpu"});
    args::Flag vf_use_octree(parser, "use_octree", "Flag to force octree backprojection (CPU only for now)", {"octree"});
    args::Flag vf_use_phasor(parser, "use_phasor", "Flag to enable phasor field reconstruction (CPU only for now)", {"phasor"});
    args::ValueFlag<float> vf_wavelength(parser, "wavelength", "Phasor field wavelength (only used with phasor enabled)", {'w', "wl", "wavelength"}, 0.005);

    try
    {
        parser.ParseCLI(argc, argv);
        filename = args::get(vf_filename);
        highest_bounce = args::get(vf_highest_bounce);
        voxel_resolution = args::get(vf_voxel_resolution);
        volume_position = args::get(vf_volume_position);
        volume_size = args::get(vf_volume_size);
        volume_direction = args::get(vf_volume_direction);
        outfile = args::get(vf_outfile);
        use_cpu = args::get(vf_use_cpu);
        use_octree = args::get(vf_use_octree);
        use_phasor_fields = args::get(vf_use_phasor);
        wavelength = args::get(vf_wavelength);
    }
    catch (const args::Completion &e)
    {
        std::cout << e.what();
        exit(1);
    }
    catch (const args::Help &)
    {
        std::cout << parser;
        exit(1);
    }
    catch (const args::ParseError &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }

    std::vector<uint32_t> bounces(0, highest_bounce - 2);
    for (int i = 3; i <= highest_bounce; i++)
        bounces.push_back(i);

    // Read dataset forcing data to be in rowMajor order
    NLOSDataset data = DatasetLoader::read_NLOS_dataset(filename, bounces, true, DataOrder::RowMajor);

    float deltaT = data.deltat[0];
    float t0 = data.t0[0];

    if (xt::any(xt::isnan(volume_position)))
        volume_position = data.hidden_volume_position;
    if (xt::any(xt::isnan(volume_size)))
        volume_size = data.hidden_volume_size;

    // If the resolution is the same in all dimensions we adjust it to the size. This won't allow for reconstructions with irregular voxels
    if (voxel_resolution[0] == voxel_resolution[1] && voxel_resolution[1] == voxel_resolution[2])
    {
        voxel_resolution = voxel_resolution * volume_size / xt::amax(volume_size)[0];
    }

    // if it is not row major: transient_data.shape() -> (channel, T, cy, cx, ly, lx)
    // if it is row major: transient_data.shape() -> (lx, ly, cx, cy, T, channel)
    xt::xarray<float> transient_data;
    
    // Squeeze channel and bounce dimensions
    if (data.capture == Confocal)
        transient_data = xt::view(data.data, xt::all(), xt::all(), 0, xt::all(), 0);
    else
        transient_data = xt::view(data.data, xt::all(), xt::all(), xt::all(), xt::all(), 0, xt::all(), 0);

    xt::xarray<float> volume;

    auto compute = use_cpu ? Compute::CPU : Compute::GPU;
    auto vol_access = use_octree ? VolumeAccess::Octree : VolumeAccess::Naive;

    if (use_phasor_fields)
    {
        xt::xarray<std::complex<float>> complex_volume = 
            bp::phasor_reconstruction(transient_data,
                                      data.camera_grid_positions,
                                      data.laser_grid_positions,
                                      data.camera_position,
                                      data.laser_position,
                                      t0, deltaT, data.capture,
                                      volume_position,
                                      volume_size,
                                      voxel_resolution,
                                      wavelength,
                                      compute, vol_access);
        volume.resize(complex_volume.shape());
        for (int i = 0; i < std::accumulate(volume.shape().begin(), volume.shape().end(), 1, std::multiplies<int>()); i++)
        {
            volume.data()[i] = std::abs(complex_volume.data()[i]);
        }
    }
    else
    {
        volume = bp::backproject(transient_data,
                                 data.camera_grid_positions,
                                 data.laser_grid_positions,
                                 data.camera_position,
                                 data.laser_position,
                                 t0, deltaT, data.capture,
                                 volume_position,
                                 volume_size,
                                 voxel_resolution,
                                 compute,
                                 data.data_order,
                                 vol_access);
    }
    
    if (outfile.size() == 0)
        outfile = filename.substr(0, filename.find(".hdf5")) + "_recon.hdf5";
    save_volume(outfile, volume);
    std::cout << "Backprojected to " << std::string(outfile) << std::endl;
    return 0;
}
