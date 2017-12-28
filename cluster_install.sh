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

# Install dependencies
export PATH=$PATH:$HOME/.local/bin
export PATH=$PATH:$WORK/llvm/bin
export CPATH=$CPATH:$WORK/llvm/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK/llvm/lib

module unload env
module load /sw/BASE/env/2017Q1-gcc-openmpi /sw/BASE/env/cuda-8.0.44_system-gcc
python3 -m pip install --user pipenv
cd $HOME/VoiceClassification
pipenv shell
pipenv install
