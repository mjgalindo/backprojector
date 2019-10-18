#ifndef ITER3D_HPP
#define ITER3D_HPP

#include <array>
#include <cmath>
#include <algorithm>

class iter3D 
{
private:
    size_t t_x = 0ul, t_y = 0ul, t_z = 0ul;
    size_t t_length_x, t_length_y, t_length_z, t_total_length;
    size_t t_current = 0ul;

public:
    inline size_t current() const { return t_current; } 
    inline size_t total_length() const { return t_total_length; }
    inline size_t x() const { return t_x; }
    inline size_t y() const { return t_y; }
    inline size_t z() const { return t_z; }

    iter3D(size_t length_x, size_t length_y, size_t length_z) : 
        t_length_x(length_x), t_length_y(length_y), t_length_z(length_z), 
        t_total_length(length_x * length_y * length_z + length_y * length_z + length_z) {}
    
    template <class D>
    iter3D(D lengths) :
        t_length_x(lengths[0]), t_length_y(lengths[1]), t_length_z(lengths[2]), 
        t_total_length(lengths[0] * lengths[1] * lengths[2]) {}

    void operator++()
    {
        t_current++;
        if (++t_z == t_length_z)
        {
            t_z = 0ul;
            if (++t_y == t_length_y)
            {
                t_y = 0ul;
                ++t_x;
                // Bounds check for x need to be handled by the user
            }
        }
    }

    void jump_to(size_t id)
    {
        t_z = id % t_length_z;
        id = id / t_length_z;
        t_y = id % t_length_y;
        id = id / t_length_y;
        t_x = id;
    }
};

#endif // ITER3D_HPP
