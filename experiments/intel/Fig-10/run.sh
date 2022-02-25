#!/bin/bash

cd ../../..;
source env.sh;
cd experiments/intel/Fig-10;
mkdir reports;

# compile mkl_sgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 

# compile blis test
gcc -g -O2 -std=c99 -Wall -Wno-unused-function -Wfatal-errors -fPIC  \
-D_POSIX_C_SOURCE=200112L -fopenmp -I${CAKE_HOME}/include/blis \
-DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o;
g++ blis_test.o $CAKE_HOME/blis/lib/haswell/libblis.a  -lm -lpthread -fopenmp \
-lrt -o blis_test;

# compile cake_sgemm_test
make;

NTRIALS=1;
NCORES=10;

# run matmul bench through intel vtune 
for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/prof_result \
		 $PWD/mkl_sgemm_test $i; 

		vtune -report summary -r prof_result -format csv \
			-report-output reports/report_mkl_$i-$j.csv -csv-delimiter comma;

		rm -rf prof_result;
	done
done


# # array=( 2 3 4 5 6 8 10 )
# # for i in "${array[@]}"
# # for j in {1..$NTRIALS}
for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/cake_sgemm_result \
			 $PWD/cake_sgemm_test $i;

		vtune -report summary -r cake_sgemm_result -format csv \
			-report-output reports/report_cake_sgemm_$i-$j.csv -csv-delimiter comma;

		rm -rf cake_sgemm_result;
	done
done



for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/blis_result \
			 $PWD/blis_test 23040 23040 23040 $i;

		vtune -report summary -r blis_result -format csv \
			-report-output reports/report_blis_$i-$j.csv -csv-delimiter comma;

		rm -rf blis_result;
	done
done


python plots.py $NTRIALS;
