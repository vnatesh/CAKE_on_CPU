#!/bin/bash

open_blas="~/Harvard/research/3DIC/amd/mm2.o";
cake="/tmp/CAKE_on_CPU/cake_dgemm_test.x";

mkdir reports;
for j in {1..10}
do
	for i in {1..16}
	do
		echo "======= Starting Cake with $i threads =========="
		AMDuProfCLI collect --event event=pmcx044,umask=0x48,interval=50000 --event event=timer --omp -o reports/report_cake_amd_$i-$j $cake $i;
		AMDuProfCLI report -i reports/report_cake_amd_$i-$j.caperf -o reports/;
		mv reports/report_cake_amd_$i-$j/report_cake_amd_$i-$j.csv reports;
		rm -rf reports/report_cake_amd_$i-$j;
		rm reports/report_cake_amd_$i-$j.caperf;
	done
done


for j in {1..10}
do
	for i in {1..16}
	do
		echo "======= Starting openblas with $i threads =========="
		AMDuProfCLI collect --event event=pmcx044,umask=0x48,interval=50000 --event event=timer --omp -o reports/report_openblas_amd_$i-$j $open_blas $i;
		AMDuProfCLI report -i reports/report_openblas_amd_$i-$j.caperf -o reports/;
		mv reports/report_openblas_amd_$i/report_openblas_amd_$i-$j.csv reports;
		rm -rf reports/report_openblas_amd_$i-$j;
		rm reports/report_openblas_amd_$i-$j.caperf;
	done
done