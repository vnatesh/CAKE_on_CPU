#!/bin/bash


# run matmul bench for 1..10 threads through intel vtune
# source /opt/intel/oneapi/setvars.sh;
# cd /home/vnatesh/CAKE_on_CPU;
mkdir bw_reports;


# compile mkl_dgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 

# compile cake_dgemm_test
make;

for j in {1..10}
do
	for i in {1..10}
	do
		/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o bw_reports/cake_socw_$i-$j -p ./cake_sgemm_test $i;
		sleep 1;
		/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o bw_reports/mkl_socw_$i-$j -p ./mkl_sgemm_test $i;
		# vtune --collect memory-access -data-limit=0 \
		# 	-result-dir=/home/vnatesh/CAKE_on_CPU/prof_result \
		#  /home/vnatesh/CAKE_on_CPU/mkl_sgemm_test $i; 

		# vtune -report summary -r prof_result -format csv \
		# 	-report-output reports/report_mkl_$i-$j.csv -csv-delimiter comma;
		# rm -rf prof_result;
	done
done


# # array=( 2 3 4 5 6 8 10 )
# # for i in "${array[@]}"
# for j in {1..10}
# do
# 	for i in {1..10}
# 	do
# 		vtune --collect memory-access -data-limit=0 \
# 			-result-dir=/home/vnatesh/CAKE_on_CPU/cake_dgemm_result \
# 			 /home/vnatesh/CAKE_on_CPU/cake_dgemm_test.x $i;

# 		vtune -report summary -r cake_dgemm_result -format csv \
# 			-report-output reports/report_cake_dgemm_$i-$j.csv -csv-delimiter comma;

# 		rm -rf cake_dgemm_result;
# 	done
# done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
# scp -r reports vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
