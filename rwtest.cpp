
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <H5Cpp.h>
#include <chrono>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xfunction.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xbuilder.hpp>

int main() 
{
    const std::string FILE_NAME = "test.hdf5";
    const std::string DS_NAME = "test";
    const int DIM_RANK = 7;

    std::vector<hsize_t> fdim = {1, 3, 3, 2, 2, 2, 2}; // dim sizes of ds (on disk)

    {
        H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
        H5::DataSpace fspace(DIM_RANK, fdim.data());
        H5::DataSet dataset(file.createDataSet(DS_NAME, H5::PredType::NATIVE_FLOAT, fspace));

        hsize_t total_elements = 1;
        for (int i = 0; i < DIM_RANK; i++)
        {
            total_elements *= fdim[i];
        }
        xt::xarray<float> buffer = xt::zeros<float>(fdim);
        for (uint i = 0; i < total_elements; i++) 
        {
            buffer.data()[i] = (float) i;
        }
        std::cout << buffer << std::endl;
        dataset.write(buffer.data(), H5::PredType::NATIVE_FLOAT);
        file.close();

        xt::xarray<float> whatWeReadLater = xt::view(buffer, xt::all(), xt::all(), 1, xt::all());
        std::cout << whatWeReadLater << std::endl;

    }

    std::cout << "FILE IS WRITTEN" << std::endl;

    {
        H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
        H5::DataSet dataset(file.openDataSet(DS_NAME));
        fdim[2] = 1; // WE ONLY READ ONE ELEMENT FROM DIMENSION 2

        hsize_t elements_to_read = 1;
        for (int i = 0; i < DIM_RANK; i++) 
        {
            elements_to_read *= fdim[i];
        }
        H5::DataSpace fspace = dataset.getSpace();

        std::vector<hsize_t> begins = {0, 0, 1, 0, 0, 0, 0};
        std::vector<hsize_t> counts = {1, 3, 1, 2, 2, 2, 2};

        fspace.selectHyperslab(H5S_SELECT_SET, counts.data(), begins.data());

        for (uint i = 0; i < begins.size(); i++) std::cout << begins[i] << ' ';
        std::cout << std::endl;
        for (uint i = 0; i < counts.size(); i++) std::cout << counts[i] << ' ';
        std::cout << std::endl;
        std::cout << std::endl;

        for (uint i = 0; i < fdim.size(); i++) std::cout << fdim[i] << ' ';
        std::cout << std::endl;

        // Defining the memory dataspace
        H5::DataSpace memspace(7, fdim.data());

        std::vector<hsize_t> mem_begins = {0, 0, 0, 0, 0, 0, 0};
        std::vector<hsize_t> mem_counts = {1, 3, 1, 2, 2, 2, 2};
        memspace.selectHyperslab(H5S_SELECT_SET, mem_counts.data(), mem_begins.data());

        std::cout << "Hyperslab is set" << std::endl;

        xt::xarray<float> buffer = xt::zeros<float>(fdim);
        dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT, memspace, fspace);
        std::cout << buffer << std::endl;

        file.close();
    }

}