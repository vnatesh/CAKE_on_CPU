#!/bin/bash

# compile mkl_sgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 

# compile cake_sgemm_test
make;

echo "algo,p,size,time" >> results_sq;


NTRIALS=1;
NCORES=10;

# run matmul bench  
for ((j=1; j <= $NTRIALS; j++));
do
	for i in {1000..3000..500}
	do
		for ((p=1; p <= $NCORES; p++));
		do
			./mkl_sgemm_test $p $i; 
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

