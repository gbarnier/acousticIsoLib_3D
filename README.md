# acousticIsoLib_3D

##DESCRIPTION
Note use cmake 3.14

##COMPILATION

To build library run:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=installation_path -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -DCMAKE_BUILD_TYPE=Debug ../acoustic_iso_lib_3d/

make install

```
