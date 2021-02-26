#!/bin/bash

# git clone https://github.com/vnatesh/CAKE_on_CPU.git
# cd CAKE_on_CPU
# make install

# Download BLIS kernels
git clone https://github.com/flame/blis.git
cd blis

# configire BLIS with openmp
./configure --enable-threading=openmp auto
make -j4
make check

# install BLIS
sudo make install
make distclean
export BLIS_INSTALL_PATH=/usr/local
cd ..

# build CAKE pytorch extension 
cd python
python3 setup.py install --user

