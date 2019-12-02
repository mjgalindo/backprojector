#ifndef NLOS_ENUMS_HPP
#define NLOS_ENUMS_HPP

namespace nlos
{
    
enum Compute
{
    CPU, 
    GPU,
    ComputeNone,
};

enum VolumeAccess
{
    Naive,
    Octree,
    VolumeAccessNone,
};

enum DataOrder
{
    ColumnMajor,
    RowMajor,
    DataOrderNone,
};

enum CaptureStrategy
{
    Exhaustive,
    Confocal,
    CaptureStrategyNone,
};

}; // namespace nlos

#endif // NLOS_ENUMS_HPP