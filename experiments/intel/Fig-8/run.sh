#!/bin/bash

cd ../../..;
source env.sh;
cd experiments/intel/Fig-8;

# compile mkl_sgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 

# compile cake_sgemm_test
make;
echo "algo,M,K,N,time" >> results_full;

NTRIALS=1;

# run matmul bench  
for ((j=1; j <= $NTRIALS; j++));
do
	for n in 1 2
	do
		for m in {500..8000..500}
		do
			for k in 1 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000
			do
				./mkl_sgemm_test $m $k $n; 
				sleep 1;
			done
		done
	done
done


for ((j=1; j <= $NTRIALS; j++));
do
	for n in 1 2
	do
		for m in {500..8000..500}
		do
			for k in 1 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000
			do
				./cake_sgemm_test $m $k $n; 
				sleep 1;
			done
		done
	done
done


for ((j=1; j <= $NTRIALS; j++));
do
	for n in 4 8
	do
		for m in {1000..8000..1000}
		do
			for k in 1 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000
			do
				./mkl_sgemm_test $m $k $n; 
				sleep 1;
			done
		done
	done
done


for ((j=1; j <= $NTRIALS; j++));
do
	for n in 4 8
	do
		for m in {1000..8000..1000}
		do
			for k in 1 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000
			do
				./cake_sgemm_test $m $k $n; 
				sleep 1;
			done
		done
	done
done


python3 plots.py $NTRIALS;