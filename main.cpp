#include "backproject.hpp"
#include "backproject_cuda.hpp"
#include "nlos_loader.hpp"

#include "args.hxx"

using namespace H5;

void save_volume(const std::string &output_file,
                 const xt::xarray<float> &volume)
{
    H5File file(output_file, H5F_ACC_TRUNC);
    auto shape = volume.shape();
    std::vector<hsize_t> fdim = {shape[0], shape[1], shape[2]};
    std::vector<hsize_t> start = {0, 0, 0};
    std::vector<hsize_t> count = {shape[0], shape[1], shape[2]};
    std::vector<hsize_t> chunks = {std::min(32ul, volume.shape()[0]), 
                                   std::min(32ul, volume.shape()[1]), 
                                   std::min(32ul, volume.shape()[2])};

    float fillvalue = NAN;
    DSetCreatPropList proplist;
    proplist.setDeflate(4);
    proplist.setFillValue(PredType::NATIVE_FLOAT, &fillvalue);
    proplist.setChunk(chunks.size(), chunks.data());

    DataSpace fspace(fdim.size(), fdim.data());
    DataSet dataset = file.createDataSet("voxelVolume", PredType::NATIVE_FLOAT, fspace, proplist);
    dataset.write(volume.data(), PredType::NATIVE_FLOAT);
}

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

    args::ValueFlag<bool> vf_use_cpu(parser, "use_cpu", "Flag to force CPU backprojection", {"cpu"}, false);

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

    NLOSData data(filename, bounces, true);

    bool is_confocal = data.is_confocal[0];
    float deltaT = data.deltat[0];
    float t0 = data.t0[0];

    if (xt::any(xt::isnan(volume_position)))
        volume_position = data.hidden_volume_position;
    if (xt::any(xt::isnan(volume_size)))
        volume_size = data.hidden_volume_size;

    if (voxel_resolution[0] == voxel_resolution[1] && voxel_resolution[1] == voxel_resolution[2])
    {
        voxel_resolution = voxel_resolution * volume_size / xt::amax(volume_size)[0];
    }

    // if it is not row major: transient_data.shape() -> (channel, T, cy, cx, ly, lx)
    // if it is row major: transient_data.shape() -> (lx, ly, cx, cy, T, channel)
    xt::xarray<float> transient_data;

    // Squeeze channel and bounce dimensions
    if (data.is_row_major)
    {
        if (is_confocal)
            transient_data = xt::view(data.data, xt::all(), xt::all(), 0, xt::all(), 0);
        else
            transient_data = xt::view(data.data, xt::all(), xt::all(), xt::all(), xt::all(), 0, xt::all(), 0);
    }
    else
    {
        transient_data = xt::view(data.data, 0, xt::all(), 0, xt::all());
    }
    
    xt::xarray<float> volume;

    if (use_cpu)
    {
        volume = bp::backproject(transient_data,
                                 data.camera_grid_positions,
                                 data.laser_grid_positions,
                                 data.camera_position,
                                 data.laser_position,
                                 t0, deltaT, is_confocal,
                                 volume_position,
                                 volume_size[0],
                                 voxel_resolution[0]);
    }
    else
    {
        volume = bp::gpu_backproject(transient_data,
                                     data.camera_grid_positions,
                                     data.laser_grid_positions,
                                     data.camera_position,
                                     data.laser_position,
                                     t0, deltaT, is_confocal,
                                     volume_position,
                                     volume_size,
                                     voxel_resolution,
                                     data.is_row_major);
    }
    if (outfile.size() == 0)
        outfile = filename.substr(0, filename.find(".hdf5")) + "_recon.hdf5";
    save_volume(outfile, volume);
    std::cout << "Backprojected to " << std::string(outfile) << std::endl;
    return 0;
}
