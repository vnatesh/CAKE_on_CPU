#!/bin/bash

# Run matmul bench on raspberry pi 3b for 1..4 cores through linux perf

# compile arm_test with ARMPL
gcc -I/opt/arm/armpl_20.3_gcc-7.1/include -fopenmp  arm_test.c -o \
test.o  /opt/arm/armpl_20.3_gcc-7.1/lib/libarmpl_lp64_mp.a \
-L{ARMPL_DIR}/lib -lm -o arm_test;

mkdir reports_arm;

# compile cake_sgemm_test
make;

NTRIALS=10;
NCORES=4;

# run matmul bench
for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		perf stat -e rC0 -o reports_arm/report_cake_$i-$j ./cake_sgemm_test $i;
	done
done



for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		perf stat -e rC0 -o reports_arm/report_arm_$i-$j ./arm_test $i;
	done
done


python3 plots.py $NTRIALS; 
