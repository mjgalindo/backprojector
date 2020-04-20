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

# Installation

Clone the repository with:

```git clone --recursive https://github.com/mjgalindo/backprojector```

to get **most** dependencies, or use:

```git submodule update --init``` 

if you cloned it normally.

## Dependencies

You will need to install CUDA, the HDF5 library and headers, python and numpy. 

Ubuntu:

```apt install hdf5-tools libhdf5-serial-dev libpython-all-dev python-numpy```

And install cuda manually from the nvidia website.

Arch:

```pacman -S hdf5 base-devel cuda```

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

By default it will compile for many relevant architectures, so you may not need to specify this.

At the time of writing, cuda 10.2 is the most recent version which will not work correctly with gcc-9.
In that case, you can solve this issue using the cmake flag `-DCMAKE_CXX_COMPILER=g++-8`.

# Usage

The ```backprojector``` binary takes a file from [our dataset](graphics.unizar.es/nlos) and backprojects the default hidden region. You can have a look at the options with the ```-h``` argument, allowing you to choose what to reconstruct.

We also prepared a simple python binding for the backprojection routine, taking numpy arrays as inputs to simplify interfacing it with other data formats.

## Example

Download a dataset from graphics.unizar.es/nlos , (we'll use [this one](https://drive.google.com/uc?export=download&id=1_niEa4nThL00Gi206d4QH2nylRa-Y7st) next).
Running ```./backprojector serapis_l[0.00,-0.50,-0.41]_r[0.00,0.00,-1.57]_v[0.82]_s[16]_l[16]_gs[1.00].hdf5``` should create a ```serapis_l[0.00,-0.50,-0.41]_r[0.00,0.00,-1.57]_v[0.82]_s[16]_l[16]_gs[1.00]_recon.hdf5``` file which you can easily load and visualize on MATLAB with:

```
v = h5read('serapis_l[0.00,-0.50,-0.41]_r[0.00,0.00,-1.57]_v[0.82]_s[16]_l[16]_gs[1.00]_recon.hdf5', 'voxelVolume');
volumeViewer(v);
```

You can have a look at acceptable flags running

```backprojector -h```

## Phasor fields reconstruction

The option `--phasor=1` (1 enables, 0 disables, same for the --cpu option) enables phasor field reconstruction using complex numbers directly, therefore avoiding doing separate backprojections for the imaginary and real components.


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


# TODO

* Check the correctness of the phasor field functionality.
* Unifying the code for phasor fields and CPU/GPU backprojection meant some performance penalty. With a good profiling, it should be possible to find the biggest bottlenecks to backprojection.
* Try octree backprojection. There's a branch specifically working with octree backprojection, where the reconstructions start very small to check regions of interest and reject empty ones. Then the resolution would increase up to a target, ideally taking much less time. 
* More efficient single point reconstructions. When the laser or the spad are focused on a single point, there's no way to reconstruct more than one point in depth (using polar coordingates instead of a regular grid for reconstruction), as there's no visibility and the first point would block the line of sight of any other points behind it. Therefore, instead of reconstructing a volume, you could theoretically end up with a single 2D depth map, where each pixel has a depth and an intensity.
* Make this repository public. There's been talks of releasing this repository to the public, letting people use it to quickly compare their reconstructions with a baseline with a relatively simple setup. That would also be an incentive to use data from the dataset.

[1] Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging, Nature Communications, 2012