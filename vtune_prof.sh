#!/bin/bash


# run matmul bench for 1..10 threads through intel vtune
# source /opt/intel/oneapi/setvars.sh;
cd /home/vnatesh/Documents/cake_pytorch/;
mkdir reports;


# compile mkl_dgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_dgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_dgemm_test; 

# compile cake_dgemm_test
make;


for i in {2..10}
do
	vtune --collect memory-access -data-limit=0 \
		-result-dir=/home/vnatesh/Documents/cake_pytorch/prof_result \
	 /home/vnatesh/Documents/cake_pytorch/mkl_dgemm_test $i; 

	vtune -report summary -r prof_result -format csv \
		-report-output reports/report_mkl_$i.csv -csv-delimiter comma;

	rm -rf prof_result;
done


# array=( 2 3 4 5 6 8 10 )
# for i in "${array[@]}"
for i in {2..10}
do
	vtune --collect memory-access -data-limit=0 \
		-result-dir=/home/vnatesh/Documents/cake_pytorch/cake_dgemm_result \
		 /home/vnatesh/Documents/cake_pytorch/cake_dgemm_test.x $i;

	vtune -report summary -r cake_dgemm_result -format csv \
		-report-output reports/report_cake_dgemm_$i.csv -csv-delimiter comma;

	rm -rf cake_dgemm_result;
done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
scp -r reports vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
