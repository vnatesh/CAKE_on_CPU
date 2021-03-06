# Overview
This repository contains an implementation of the CAKE algorithm for CPUs. It utilizes the BLIS microkernel in the innermost loop of MMM. The microkernel makes use of SIMD processing elements (PEs) in the CPU whereas CAKE was originally designed with systolic array PEs in mind.

## Installation

```bash
git clone https://github.com/vnatesh/CAKE_on_CPU.git
cd CAKE_on_CPU
make install
source env.sh
```

Installation automatically downloads and installs the following tool/dependency verions:

* `BLIS` 


## Usage
See [wiki](https://github.com/vnatesh/CAKE_on_CPU/wiki) for usage.

<!-- <p align = "center">
<img  src="https://github.com/vnatesh/maestro/blob/master/images/cake_diagram.png" width="500">
</p>
 -->