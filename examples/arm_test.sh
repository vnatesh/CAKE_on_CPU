#!/bin/bash

gcc -I/opt/arm/armpl_20.3_gcc-7.1/include -fopenmp  arm_test.c -o \
test.o  /opt/arm/armpl_20.3_gcc-7.1/lib/libarmpl_lp64_mp.a \
-L{ARMPL_DIR}/lib -lm -o arm_test;

./arm_test 4;
