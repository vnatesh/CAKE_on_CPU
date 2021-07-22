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


# array=( 2 3 4 5 6 8 10 )
# for i in "${array[@]}"
# for j in {1..$NTRIALS}
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


python plots.py $NTRIALS;
