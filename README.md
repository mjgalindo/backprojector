# Backprojector: NLOS reconstructions on the GPU

Backprojector is a CUDA implementation of the backprojection routine used for NLOS imaging in the work of Velten and colleagues [1].

We created this code as part of our work in ["A Dataset for Benchmarking Time-Resolved Non-Line-of-Sight Imaging"](graphics.unizar.es/nlos). You are free to use it in your own work, just please cite us:

```{bibtex}
@misc{galindo19-NLOSDataset,
    title={Zaragoza NLOS synthetic Dataset},
    author={Galindo, Miguel and Marco, Julio and O'Toole, Matthew and Wetzstein, Gordon and Gutierrez, Diego and
    Jarabo, Adrian},
    url={https://graphics.unizar.es/nlos},
    year={2019},
}
```

We also accept requests on new features or pull-requests on fixes or improvements, so if you want to contribute you know what to do.

# Install

Clone the repository with:

```git clone --recursive https://github.com/mjgalindo/backprojector```

to get most dependencies, or use:

```git submodule update --init``` 

if you cloned it normally.

## Dependencies

You will need to install CUDA, the HDF5 library and headers, python and numpy. 

On ubuntu:

```apt install hdf5-tools libhdf5-serial-dev libpython-all-dev python-numpy```

You should be able to find equivalent packages in the repositories for your system or on the corresponding websites on windows (note that while this project has been successfully compiled for windows, we don't work directly on that platform so instability is to be expected).

To compile for your specific system, look for your GPU in this website https://developer.nvidia.com/cuda-gpus and take a note of its compute capability (without the dot). Then just compile like this on a linux system:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_COMPUTE_CAPABILITY=XX ..
make -j
# Optionally, you can install it with
sudo make install
```

replacing XX with the previous value.

# Usage

The ```backprojector``` binary takes a file from [our dataset](graphics.unizar.es/nlos) and backprojects the default hidden region. You can have a look at the options with the ```-h``` argument, allowing you to choose what to reconstruct.

We also prepared a simple python binding for the backprojection routine, taking numpy arrays as inputs to simplify interfacing it with other data formats.

# Performance

In general, we've seen the GPU implementation achieve around a 5-10x speed-up over a multithreaded CPU one.

The following table was measured on an NVIDIA Quadro P5000:

|Captured grid|Volume resolution|Rec. Time|Expected FPS|
|-------------|-----------------|----|------------|
|32x32| 8x8x8| 5ms |200fps|
|32x32| 32x32x32| 13ms| 79fps|
|64x64| 8x8x8| 12ms| 83fps|
|64x64| 32x32x32| 37ms| 27fps|
|64x64| 64x64x64| 213ms| 4.5fps|
|128x128| 8x8x8| 34ms| 29fps|
|128x128| 32x32x32| 170ms| 5.8fps|
|128x128| 64x64x64| 859ms| 1.1fps|
|128x128| 128x128x128| 5993ms| 0.16fps|



[1] Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging, Nature Communications, 2012