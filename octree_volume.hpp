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
        m_data = xt::zeros<T>(m_max_voxels);
    }

    octree_volume(xt::xarray<float>& volume,
                  const xt::xarray<float>& volume_size,
                  const xt::xarray<float>& volume_position) 
    {
        m_max_voxels = xt::xarray<size_t>(volume.shape());
        m_volume_size = volume_size;
        m_volume_position = volume_position;
        m_voxel_size = volume_size / m_max_voxels;
        m_data = volume;
    }

    xt::xarray<T>& volume()
    {
        return m_data;
    }

    size_t total_voxels(size_t depth = -1) const
    {
        return xt::prod(m_max_voxels)[0];
    }

    uint32_t max_depth() const
    {
        return xt::amin(xt::log2(m_max_voxels))[0];
    }

    template <typename UT>
    T& operator()(const xt::xarray<UT>& xyz)
    {
        assert(xt::sum(xyz * m_max_voxels)[0] < (*this).total_voxels());
        return m_data(xyz[0], xyz[1], xyz[2]);
    }

    template <typename UT>
    T& operator()(const xt::xarray<UT>& xyz, int depth)
    {
        assert(xt::sum(xyz * m_max_voxels)[0] < (*this).total_voxels());
        return (*this)((*this).at_depth(xyz, depth));
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

    xt::xarray<T> position_at(const iter3D& iter) const
    {
        static xt::xarray<T> zero_pos = m_volume_position - m_volume_size / 2;
        return std::move(zero_pos + xt::xarray<T>{iter.x() * m_voxel_size[0], iter.y() * m_voxel_size[1], iter.z() * m_voxel_size[2]});
    }

    xt::xarray<T> position_at(const iter3D& iter, int depth) const
    {
        static xt::xarray<T> zero_pos = m_volume_position - m_volume_size / 2;
        return std::move(zero_pos + xt::xarray<T>{iter.x() * m_voxel_size[0], iter.y() * m_voxel_size[1], iter.z() * m_voxel_size[2]});
    }

    xt::xarray<size_t> max_voxels() const
    {
        return m_max_voxels;
    }

    template <typename TI>
    inline xt::xarray<TI> at_depth(const xt::xarray<TI>& xyz, int depth) const
    {
        xt::xarray<TI> mod = m_max_voxels / (1u << depth);
        return xyz / mod;
    }

};

#endif // OCTREEVOL_HPP
