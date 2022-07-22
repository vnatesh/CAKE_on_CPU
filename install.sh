#!/bin/bash

# git clone https://github.com/vnatesh/CAKE_on_CPU.git
# cd CAKE_on_CPU
# make install

# Download BLIS kernels

if uname -m | grep -q 'aarch64'; 
then
   python3 python/kernel_gen.py armv8 20 72
   mv sparse.cpp dense.cpp src/kernels/armv8
else
   python3 python/kernel_gen.py haswell 20 96
   mv sparse.cpp dense.cpp src/kernels/haswell
fi


if [ "$1" == "blis" ]; 
then

	git clone https://github.com/amd/blis.git

	BLIS_PATH=$PWD
	cd blis

	# reset to older blis version for now
	#git reset --hard 961d9d5

	# ./configure CC=aarch64-linux-gnu-gcc --prefix=$BLIS_PATH --enable-threading=openmp cortexa53
	# install BLIS in curr dir and configire with openmp
	./configure --prefix=$BLIS_PATH --enable-threading=openmp auto
	# ./configure --enable-threading=openmp haswell
	make -j4
	make check

	# install BLIS
	make install
	#make distclean
	cd ..
fi
#source ./env.sh
#make build

# # build CAKE pytorch extension 
# cd python
# python3 setup.py install --user

