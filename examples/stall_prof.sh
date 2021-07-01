#!/bin/bash


# run matmul bench for 1..10 threads through intel vtune
# source /opt/intel/oneapi/setvars.sh;
cd /home/vnatesh/CAKE_on_CPU/examples;
mkdir report_stalls;


# compile mkl_dgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 

# compile cake_sgemm_test
make;


for j in {1..2}
do
	for m in 10 50 100 1000 2000 3000
	do
		for k in 10 50 100 1000 2000 3000
		do
			vtune --collect memory-access -knob sampling-interval=0.01 -data-limit=0 \
				-result-dir=/home/vnatesh/CAKE_on_CPU/examples/prof_result \
			 /home/vnatesh/CAKE_on_CPU/examples/mkl_sgemm_test $m $k; 

			vtune -report summary -r prof_result -format csv \
				-report-output report_stalls/report_mkl_$m-$k-$j.csv -csv-delimiter comma;

			rm -rf prof_result;
		done
	done
done


# array=( 2 3 4 5 6 8 10 )
# for i in "${array[@]}"
for j in {1..2}
do
	for m in 10 50 100 1000 2000 3000
	do
		for k in 10 50 100 1000 2000 3000
		do
			vtune --collect memory-access -knob sampling-interval=0.01 -data-limit=0 \
				-result-dir=/home/vnatesh/CAKE_on_CPU/examples/cake_sgemm_result \
				 /home/vnatesh/CAKE_on_CPU/examples/cake_sgemm_test $m $k;

			vtune -report summary -r cake_sgemm_result -format csv \
				-report-output report_stalls/report_cake_sgemm_$m-$k-$j.csv -csv-delimiter comma;

			rm -rf cake_sgemm_result;
		done
	done
done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
# scp -r report_stalls vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
