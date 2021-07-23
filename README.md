# Overview
This repository contains an implementation of the CAKE algorithm for CPUs. It utilizes the BLIS microkernel in the innermost loop of MMM. The microkernel makes use of SIMD processing elements (PEs) in the CPU whereas CAKE was originally designed with systolic array PEs in mind.

## Installation

```bash
git clone https://github.com/vnatesh/CAKE_on_CPU.git
cd CAKE_on_CPU
make install
source env.sh
make build
```

Installation automatically downloads and installs the following tool/dependency verions:

* `BLIS` 


## Quick Start

In the `examples` directory, you will find a simple script `cake_sgemm_test.cpp` that performs CAKE matrix multiplication on random input matrices given M, K, and N values as command line arguments. To compile the script, simple type `make` and run the script as shown below. Make sure you have sourced the `env.sh` file before running. 

```bash
~/CAKE_on_CPU/examples$ make
g++ -I/home/vnatesh/CAKE_on_CPU/include -I/usr/local/include/blis cake_sgemm_test.cpp -L/home/vnatesh/CAKE_on_CPU -lcake -o cake_sgemm_test

~/CAKE_on_CPU/examples$ ./cake_sgemm_test 3 3 3
M = 3, K = 3, N = 3
0.546852	0.546852
-0.430778	-0.430778
-0.633527	-0.633527
-0.433842	-0.433842
0.640107	0.640107
0.383761	0.383761
-0.208048	-0.208048
-0.454641	-0.454641
1.107274	1.107274
CORRECT!
```

## Running Experiments:

Before running experiments, make sure the following additional dependencies are installed:

* `Intel`
	* `Vtune 2021.1.1` 
	* `OpenMP 4.5` 
	* `Linux perf 5.4.86` 

* `AMD` 
	* `OpenMP 4.5` 
	* `AMD uProf 3.4.468` 
	* `OpenBLAS 0.3.14.dev` 

* `ARM` 
	* `ARMPL 21.0.0` 
	* `OpenMP 4.5` 
	* `Linux perf 5.4.86` 

The experiments are organized in separate directories for each CPU architecture tested (Intel, AMD, ARM). Each arch-specific directory contains sub-directories corresponding to figures in the CAKE paper (<http://www.eecs.harvard.edu/~htk/publication/2021-sc-kung-natesh-sabot.pdf>). To run an experiment and plot the associated figure, simply enter the directory and execute the `run.sh` file. An example to generate Figure 10 for the Intel CPU tested is shown below. Experiments should be performed in `sudo` mode to enable permissions for hardware profiling.

```bash
~/CAKE_on_CPU$ sudo -s
~/CAKE_on_CPU$ source env.sh
~/CAKE_on_CPU$ cd experiments/intel/Fig-10
~/CAKE_on_CPU/experiments/intel/Fig-10$ ./run.sh
```

## Details
See [wiki](https://github.com/vnatesh/CAKE_on_CPU/wiki) for more details.

<!-- <p align = "center">
<img  src="https://github.com/vnatesh/maestro/blob/master/images/cake_diagram.png" width="500">
</p>
 -->


