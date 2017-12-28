#!/bin/bash

# Install LLVM
cd $WORK
wget http://releases.llvm.org/5.0.0/llvm-5.0.0.src.tar.xz
tar xf llvm-5.0.0.src.tar.xz
cd llvm-5.0.0.src
mkdir build && cd build
module load cmake
cmake ../
cmake --build .
cmake -DCMAKE_INSTALL_PREFIX=$WORK/llvm -P cmake_install.cmake

# Install Python dependencies
export PATH=$PATH:$HOME/.local/bin
export PATH=$PATH:$WORK/llvm/bin
export CPATH=$CPATH:$WORK/llvm/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK/llvm/lib

module unload env; module load /sw/BASE/env/2017Q1-gcc-openmpi /sw/BASE/env/cuda-8.0.44_system-gcc
wget https://pypi.python.org/packages/e3/d3/4a356db5b6a2c9dcb30011280bc065cf51de1e4ab5a5fee44eb460a98449/tensorflow_gpu-1.4.1-cp36-cp36m-manylinux1_x86_64.whl#md5=b5ceb9705a8bbf57d854266a882c5757
python3 -m pip install --user tensorflow_gpu-1.4.1-cp36-cp36m-manylinux1_x86_64.whl
python3 -m pip install --user --upgrade pip librosa pandas keras
