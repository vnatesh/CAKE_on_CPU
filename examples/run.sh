#!/bin/bash



gcc -g -O2 -std=c99 -Wall -Wno-unused-function -Wfatal-errors -fPIC  \
-D_POSIX_C_SOURCE=200112L -fopenmp -I${CAKE_HOME}/include/blis \
-DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o;
g++ blis_test.o $CAKE_HOME/blis/lib/haswell/libblis-mt.a  -lm -lpthread -fopenmp \
-lrt -o blis_test;


make;

echo "algo,p,M,K,N,time" >> results

export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";


./cake_sgemm_test 20 10 288 4512 192
./blis_test 20 10 288 4512 192