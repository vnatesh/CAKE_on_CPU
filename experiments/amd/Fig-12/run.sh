#!/bin/bash
#!/bin/bash

cd ../../..;
source env.sh;
cd experiments/amd/Fig-12;
mkdir reports;

sudo ldconfig $CAKE_HOME;

# compile openblas test
gcc -fopenmp -o openblas openblas.c \
-I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib \
-lopenblas -lpthread -lm -ldl -lgfortran;

# compile cake_sgemm_test
make;

NTRIALS=10;
NCORES=16;

# run matmul bench through AMDuProf
for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		echo "======= Starting Cake with $i threads =========="
		AMDuProfCLI collect --event event=pmcx044,umask=0x48,interval=50000 --event event=timer --omp -o reports/report_cake_amd_$i-$j ./cake_sgemm_test $i;
		AMDuProfCLI report -i reports/report_cake_amd_$i-$j.caperf -o reports/;
		mv reports/report_cake_amd_$i-$j/report_cake_amd_$i-$j.csv reports;
		rm -rf reports/report_cake_amd_$i-$j;
		rm reports/report_cake_amd_$i-$j.caperf;
	done
done


for ((j=1; j <= $NTRIALS; j++));
do
	for ((i=1; i <= $NCORES; i++));
	do
		echo "======= Starting openblas with $i threads =========="
		AMDuProfCLI collect --event event=pmcx044,umask=0x48,interval=50000 --event event=timer --omp -o reports/report_openblas_amd_$i-$j ./openblas $i;
		AMDuProfCLI report -i reports/report_openblas_amd_$i-$j.caperf -o reports/;
		mv reports/report_openblas_amd_$i-$j/report_openblas_amd_$i-$j.csv reports;
		rm -rf reports/report_openblas_amd_$i-$j;
		rm reports/report_openblas_amd_$i-$j.caperf;
	done
done


python3 plots.py $NTRIALS;