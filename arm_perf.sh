#!/bin/bash

# Run matmul bench on raspberry pi 3b for 1..4 cores through linux perf
cd ~;
mkdir reports_arm;


# compile arm_test with ARMPL
./arm_test.sh;

# compile cake_sgemm_test
cd CAKE_on_CPU;
make;


for i in {1..4}
do
	perf stat -e rC0 -o ../reports_arm/report_cake_$i ./cake_dgemm_test.x $i;
done

cd ..;

for i in {1..4}
do
	perf stat -e rC0 -o reports_arm/report_arm_$i ./arm_test $i;
done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
scp -r reports_arm vikas@10.0.0.185:/Users/vikas/test/;
