#ifndef OCTREEVOL_HPP
#define OCTREEVOL_HPP

#include <array>
#include <cmath>
#include <algorithm>
#include <H5Cpp.h>
#include <xtensor/xarray.hpp>

#include "iter3D.hpp"

/**
 * Represents a 3D grid volume.
 * It can optionally contain a double buffer that
 * can be accessed at different depths as if it were an octree.
 */ 
template <typename DT, typename FT>
class OctreeVolume
{

protected:
    xt::xarray<size_t> m_max_voxels;
    xt::xarray<FT> m_volume_size;
    xt::xarray<FT> m_volume_position;
    xt::xarray<FT> m_voxel_size;

    xt::xarray<DT> m_data;
    xt::xarray<DT> m_data_buffer;
    bool m_double_buffered = false;

public:
    enum BuffType {Main, Buffer};

    OctreeVolume(const xt::xarray<size_t>& voxel_axes,
                 const xt::xarray<FT>& volume_size,
                 const xt::xarray<FT>& volume_position,
                 bool double_buffered=false) 
    {
        m_max_voxels = voxel_axes;
        m_volume_size = volume_size; 
        m_volume_position = volume_position;
        m_voxel_size = volume_size / voxel_axes;
        m_data = xt::zeros<DT>(m_max_voxels);
        m_double_buffered = double_buffered;
        if (m_double_buffered)
        {
            m_data_buffer = xt::zeros<DT>(m_data.shape());
        }
    }

    OctreeVolume(xt::xarray<DT>& volume,
                  const xt::xarray<FT>& volume_size,
                  const xt::xarray<FT>& volume_position,
                  bool double_buffered=false) 
    {
        auto vshape = volume.shape();
        m_max_voxels = xt::xarray<size_t>({vshape[0], vshape[1], vshape[2]});
        m_volume_size = volume_size;
        m_volume_position = volume_position;
        m_voxel_size = volume_size / m_max_voxels;
        m_data = volume;
        m_double_buffered = double_buffered;
        if (m_double_buffered)
        {
            m_data_buffer = xt::zeros<DT>(m_data.shape());
        }
    }

    void reset_buffer()
    {
        m_double_buffered = true;
        m_data_buffer = xt::zeros<DT>(m_data.shape());
    }

    void swap_buffers()
    {
        assert(m_double_buffered);
        if (m_double_buffered)
        std::swap(m_data, m_data_buffer);
    }

    xt::xarray<DT>& volume(BuffType buff = BuffType::Main)
    {
        switch (buff)
        {
            case Main:
                return m_data;
            case Buffer:
                return m_data_buffer;
            default:
                return m_data;
        }
    }

    size_t total_voxels(size_t depth) const
    {
        return xt::prod(m_max_voxels)[0];
    }

    uint32_t max_depth() const
    {
        return xt::amax(xt::log2(m_max_voxels))[0];
    }

    template <typename UT>
    DT& operator()(const xt::xarray<UT>& xyz, BuffType buff = BuffType::Main)
    {
        switch (buff)
        {
            case Main:
                return m_data(xyz[0], xyz[1], xyz[2]);
            case Buffer:
                return m_data_buffer(xyz[0], xyz[1], xyz[2]);
            default:
                return m_data(xyz[0], xyz[1], xyz[2]);
        }
    }

    template <typename UT>
    DT& operator()(const xt::xarray<UT>& xyz, int depth, BuffType buff = BuffType::Main)
    {
        return (*this)((*this).at_depth(xyz, depth), buff);
    }

    DT& operator()(const Iter3D& iter, BuffType buff = BuffType::Main)
    {   
        return (*this)(iter.x(), iter.y(), iter.z(), buff);
    }

    DT& operator()(const Iter3D& iter, int depth, BuffType buff = BuffType::Main)
    {
        return (*this)((*this).at_depth(xt::xarray<size_t>{iter.x(), iter.y(), iter.z()}, depth), buff);
    }

    template <typename UT>
    DT& operator()(UT x, UT y, UT z, BuffType buff = BuffType::Main)
    {
        return (*this)(xt::xarray<UT>({x,y,z}), buff);
    }

    template <typename UT>
    DT& operator()(UT x, UT y, UT z, int depth, BuffType buff = BuffType::Main)
    {   
        return (*this)((*this).at_+depth(xt::xarray<UT>({x,y,z}), depth, buff));
    }

    xt::xarray<FT> position_at(const Iter3D& iter) const
    {
        static xt::xarray<FT> zero_pos = m_volume_position - m_volume_size / 2;
        return std::move(zero_pos + xt::xarray<FT>{iter.x() * m_voxel_size[0], 
                                                   iter.y() * m_voxel_size[1], 
                                                   iter.z() * m_voxel_size[2]});
    }

    xt::xarray<FT> voxel_size_at(int depth=-1) const
    {
        return m_volume_size / max_voxels(depth);
    }

    xt::xarray<FT> position_at(const Iter3D& iter, int depth) const
    {
        static xt::xarray<FT> zero_pos = m_volume_position - m_volume_size / 2;
        xt::xarray<FT> voxel_size = voxel_size_at(depth);
        return std::move(zero_pos + xt::xarray<FT>{iter.x() * voxel_size[0], 
                                                   iter.y() * voxel_size[1], 
                                                   iter.z() * voxel_size[2]});
    }

    xt::xarray<size_t> max_voxels(int depth=-1) const
    {
        if (depth == (int) max_depth() || depth == -1) return m_max_voxels;
        depth = depth < 0 ? max_depth() : depth;
        xt::xarray<size_t> mod = m_max_voxels / (1u << depth);
        return m_max_voxels / mod;
    }

    template <typename UT>
    inline xt::xarray<UT> at_depth(const xt::xarray<UT>& xyz, int depth) const
    {
        if (depth == -1 || depth == (int) max_depth()) return xyz;
        xt::xarray<UT> mod = m_max_voxels / (1u << depth);
        return xyz / mod;
    }

};

template <typename FT>
using ComplexOctreeVolume = OctreeVolume<std::complex<FT>, FT>;

using ComplexOctreeVolumeF = OctreeVolume<std::complex<float>, float>;

template <typename DT>
using OctreeVolumeF = OctreeVolume<DT, float>;

using SimpleOctreeVolume = OctreeVolume<float, float>;

template <typename DT>
using OctreeVolumeD = OctreeVolume<DT, double>;


static void save_volume(const std::string &output_file,
                        const xt::xarray<float> &volume,
                        const std::string &dataset_name="voxelVolume")
{
    H5::H5File file(output_file, H5F_ACC_TRUNC);
    auto shape = volume.shape();
    std::vector<hsize_t> fdim = {shape[0], shape[1], shape[2]};
    std::vector<hsize_t> start = {0, 0, 0};
    std::vector<hsize_t> count = {shape[0], shape[1], shape[2]};
    std::vector<hsize_t> chunks = {std::min(32ul, volume.shape()[0]), 
                                   std::min(32ul, volume.shape()[1]), 
                                   std::min(32ul, volume.shape()[2])};

    float fillvalue = NAN;
    H5::DSetCreatPropList proplist;
    proplist.setDeflate(4);
    proplist.setFillValue(H5::PredType::NATIVE_FLOAT, &fillvalue);
    proplist.setChunk(chunks.size(), chunks.data());

    H5::DataSpace fspace(fdim.size(), fdim.data());
    H5::DataSet dataset = file.createDataSet(dataset_name, H5::PredType::NATIVE_FLOAT, fspace, proplist);
    dataset.write(volume.data(), H5::PredType::NATIVE_FLOAT);
}

#endif // OCTREEVOL_HPP
