#ifndef NLOS_ENUMS_HPP
#define NLOS_ENUMS_HPP

namespace nlos
{
    
enum Compute
{
    CPU, 
    GPU,
};

enum VolumeAccess
{
    Naive,
    Octree,
};

enum DataOrder
{
    ColumnMajor,
    RowMajor,
};

enum CaptureStrategy
{
    Exhaustive,
    Confocal,
};

}; // namespace nlos

#endif // NLOS_ENUMS_HPP