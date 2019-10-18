#ifndef OCTREEVOL_HPP
#define OCTREEVOL_HPP

#include <array>
#include <cmath>
#include <algorithm>


#include "iter3D.hpp"

template <typename T>
class octree_volume
{

protected:
    xt::xarray<size_t> m_max_voxels;
    xt::xarray<T> m_volume_size;
    xt::xarray<T> m_volume_position;
    xt::xarray<T> m_voxel_size;

    xt::xarray<T> m_data;

public:

    octree_volume(const xt::xarray<uint32_t>& voxel_axes,
                  const xt::xarray<float>& volume_size,
                  const xt::xarray<float>& volume_position) 
    {
        m_max_voxels = voxel_axes;
        m_volume_size = volume_size; 
        m_volume_position = volume_position;
        m_voxel_size = volume_size / voxel_axes;
        m_data.resize(m_max_voxels);
    }

    xt::xarray<T> volume()
    {
        return m_data;
    }

    size_t total_voxels(size_t depth = -1)
    {
        return xt::prod(m_max_voxels)[0];
    }

    uint32_t max_depth()
    {
        return xt::amin(xt::log2(m_max_voxels))[0];
    }

    template <typename UT>
    T& operator()(const xt::xarray<UT>& xyz)
    {
        assert(xt::sum(xyz * m_max_voxels)[0] < (*this).total_voxels());
        return m_data(xyz[0], xyz[1], xyz[2]);
        // TODO: Direct access "may" be faster?
        // return m_data.data[x * m_max_voxels[1]*  m_max_voxels[2] + y * m_max_voxels[2] + z];
    }

    template <typename UT>
    T& operator()(const xt::xarray<UT>& xyz, int depth)
    {
        assert(xt::sum(xyz * m_max_voxels)[0] < (*this).total_voxels());
        return (*this)((*this).at_depth(xyz, depth));
        // TODO: Direct access "may" be faster?
        // return m_data.data[x * m_max_voxels[1]*  m_max_voxels[2] + y * m_max_voxels[2] + z];
    }

    T& operator()(const iter3D& iter)
    {   
        return (*this)(iter.x(), iter.y(), iter.z());
    }

    T& operator()(const iter3D& iter, int depth)
    {   
        return (*this)((*this).at_depth(xt::xarray<size_t>{iter.x(), iter.y(), iter.z()}, depth));
    }

    template <typename UT>
    T& operator()(UT x, UT y, UT z)
    {
        return (*this)(xt::xarray<UT>({x,y,z}));
    }

    template <typename UT>
    T& operator()(UT x, UT y, UT z, int depth)
    {   
        return (*this)((*this).at_depth(xt::xarray<UT>({x,y,z}), depth));
    }

    xt::xarray<T> position_at(const iter3D& iter)
    {
        static xt::xarray<T> zero_pos = m_volume_position - m_volume_size / 2;
        return std::move(zero_pos + xt::xarray<T>{iter.x() * m_voxel_size[0], iter.y() * m_voxel_size[1], iter.z() * m_voxel_size[2]});
    }

    xt::xarray<T> position_at(const iter3D& iter, int depth)
    {
        static xt::xarray<T> zero_pos = m_volume_position - m_volume_size / 2;
        return std::move(zero_pos + xt::xarray<T>{iter.x() * m_voxel_size[0], iter.y() * m_voxel_size[1], iter.z() * m_voxel_size[2]});
    }

    xt::xarray<size_t> max_voxels()
    {
        return m_max_voxels;
    }

    xt::xarray<T> at_depth(xt::xarray<size_t> xyz, int depth)
    {
        xt::xarray<T> mod = m_max_voxels / (1u << depth);
        return xyz / mod;
    }

};

#endif // OCTREEVOL_HPP