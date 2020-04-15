#ifndef ITER3D_HPP
#define ITER3D_HPP

#include <array>
#include <cmath>
#include <algorithm>

/**
 * Iterator in 3D for volumes.
 * Automatically sets indices for a volume with a given 
 * shape and 
 */
class Iter3D 
{
private:
    size_t m_x = 0ul, m_y = 0ul, m_z = 0ul;
    size_t m_length_x, m_length_y, m_length_z, m_total_length;
    size_t m_current = 0ul;

public:
    inline size_t current() const { return m_current; } 
    inline size_t total_length() const { return m_total_length; }
    inline size_t x() const { return m_x; }
    inline size_t y() const { return m_y; }
    inline size_t z() const { return m_z; }

    Iter3D(size_t length_x, size_t length_y, size_t length_z) : 
        m_length_x(length_x), m_length_y(length_y), m_length_z(length_z), 
        m_total_length(length_x * length_y * length_z + length_y * length_z + length_z) {}
    
    template <class D>
    Iter3D(D lengths) :
        m_length_x(lengths[0]), m_length_y(lengths[1]), m_length_z(lengths[2]), 
        m_total_length(lengths[0] * lengths[1] * lengths[2]) {}

    void operator++()
    {
        m_current++;
        if (++m_z == m_length_z)
        {
            m_z = 0ul;
            if (++m_y == m_length_y)
            {
                m_y = 0ul;
                ++m_x;
                // Bounds check for x need to be handled by the user
            }
        }
    }

    void jump_to(size_t global_id)
    {
        m_z = global_id % m_length_z;
        global_id = global_id / m_length_z;
        m_y = global_id % m_length_y;
        global_id = global_id / m_length_y;
        m_x = global_id;
        m_current = global_id;
    }

    void jump_to(size_t x, size_t y, size_t z)
    {
        m_x = x;
        m_y = y;
        m_z = z;
        m_current = x * m_length_y * m_length_z + 
                    y * m_length_z + 
                    z;
    }
};

#endif // ITER3D_HPP
