#!/bin/bash


# ./mkl_sgemm_test 10 3000; sleep 1; ./mkl_sgemm_test 10 3000; sleep 1; ./mkl_sgemm_test 10 3000; 
# ./mkl_sgemm_test 10 3000; sleep 1; ./mkl_sgemm_test 10 3000; sleep 1; ./mkl_sgemm_test 10 3000; 
# ./mkl_sgemm_test 10 3000; sleep 1; ./mkl_sgemm_test 10 3000; sleep 1; ./mkl_sgemm_test 10 3000; 


# # ./cake_sgemm_test 10 3000; sleep 1; ./cake_sgemm_test 10 3000; sleep 1; ./cake_sgemm_test 10 3000; 
# # ./cake_sgemm_test 10 3000; sleep 1; ./cake_sgemm_test 10 3000; sleep 1; ./cake_sgemm_test 10 3000; 
# # ./cake_sgemm_test 10 3000; sleep 1; ./cake_sgemm_test 10 3000; sleep 1; ./cake_sgemm_test 10 3000; 



# run matmul bench for 1..10 threads through intel vtune
# source /opt/intel/oneapi/setvars.sh;
cd /home/vnatesh/CAKE_on_CPU/examples;
# mkdir reports;


# compile mkl_dgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 

# compile cake_sgemm_test
make;

echo "algo,p,size,time" >> results_sq;

for j in {1..10}
do
	for i in {1000..3000..500}
	do
		for p in {1..10}
		do
			./mkl_sgemm_test $p $i; 
			sleep 1;
		done
	done
done


# array=( 2 3 4 5 6 8 10 )
# for i in "${array[@]}"
for j in {1..10}
do
	for i in {1000..3000..500}
	do
		for p in {1..10}
		do
			./cake_sgemm_test $p $i;
			sleep 1;
		done
	done
done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
# scp -r reports vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
