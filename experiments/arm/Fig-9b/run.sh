#!/bin/bash

# compile arm_test with ARMPL
gcc -I/opt/arm/armpl_20.3_gcc-7.1/include -fopenmp  arm_test.c -o \
test.o  /opt/arm/armpl_20.3_gcc-7.1/lib/libarmpl_lp64_mp.a \
-L{ARMPL_DIR}/lib -lm -o arm_test;

# compile cake_sgemm_test
make;

echo "algo,p,size,time" >> results_sq;


NTRIALS=10;
NCORES=4;

# run matmul bench  
for ((j=1; j <= $NTRIALS; j++));
do
	for i in {1000..3000..500}
	do
		for ((p=1; p <= $NCORES; p++));
		do
			./arm_test $p $i; 
			sleep 1;
		done
	done
done


# array=( 2 3 4 5 6 8 10 )
# for i in "${array[@]}"
for ((j=1; j <= $NTRIALS; j++));
do
	for i in {1000..3000..500}
	do
		for ((p=1; p <= $NCORES; p++));
		do
			./cake_sgemm_test $p $i;
			sleep 1;
		done
	done
done


python plots.py $NTRIALS;






