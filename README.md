# toyDNS

![](https://github.com/humhang/toyDNS/blob/main/blast.gif?raw=true)

an toy code for 1D, 2D and 3D compressible flow simulations with scalar transport on shared memory computer architecture.
Kokkos is used for the shared memory parallelization, the code is able to run on any Kokkos backend, OpenMP/C++ Thread/CUDA/HIP/SYCL.

## dependency
toyDNS depends on Kokkos (any backend) and HDF5 for visualization purpose, if cmake failed to find these package, 
please set environment variables Kokkos_ROOT to the directory where KokkosConfig.cmake can be found.

## compiling
mkdir build && cd build && cmake .. && make && make install

## running
cd ../bin
./toyDNS

