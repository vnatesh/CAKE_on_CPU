#!/bin/bash


# compile blis test
gcc -g -O2 -std=c99 -Wall -Wno-unused-function -Wfatal-errors -fPIC  \
-D_POSIX_C_SOURCE=200112L -fopenmp -I${CAKE_HOME}/include/blis \
-DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o;
g++ blis_test.o $CAKE_HOME/blis/lib/cortexa53/libblis.a  -lm -lpthread -fopenmp \
-lrt -o blis_test;

# compile armpl test
gcc -I/opt/arm/armpl_21.0_gcc-10.2/include -fopenmp  arm_test.c -o test.o /opt/arm/armpl_21.0_gcc-10.2/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib -lm -o arm_test;

# compile cake_sgemm_test
make;


echo "algo,p,M,K,N,time" >> results

NTRIALS=1000;
NCORES=4;

./arm_test 1000
./blis_test 1000
./cake_sgemm_test 1000

# for ((j=1; j <= $NTRIALS; j++));
# do
# 	for ((i=1; i <= $NCORES; i++));
# 	do
# 		vtune --collect memory-access -data-limit=0 \
# 			-result-dir=$PWD/blis_result \
# 			 $PWD/blis_test 23040 23040 23040 $i;

# 		vtune -report summary -r blis_result -format csv \
# 			-report-output reports/report_blis_$i-$j.csv -csv-delimiter comma;

# 		rm -rf blis_result;
# 	done
# done


