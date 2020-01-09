#ifndef SPACE3D_HPP
#define SPACE3D_HPP

#include <xtensor/xarray.hpp>


template <typename DT, typename FT>
class Space3D 
{
    xt::xarray<DT> m_data;

    xt::xarray<FT> m_volume_size;
    xt::xarray<FT> m_volume_position;

    Space3D(const xt::xarray<FT> volume_size,
            const xt::xarray<FT> volume_position) :
        m_volume_size(volume_size); 
        m_volume_position(volume_position) {}

    Space3D(const xt::xarray<DT> space, 
            const xt::xarray<FT> volume_size,
            const xt::xarray<FT> volume_position)
        m_data(space),
        m_volume_size(volume_size); 
        m_volume_position(volume_position) {}

};