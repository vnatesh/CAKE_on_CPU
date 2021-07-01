import os

M=500
K=1500
N=500

# compile mkl_dgemm_test with Intel MKL
os.system('''gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test''')

# compile cake_sgemm_test
os.system("make")
os.system('''echo "algo,M,K,N,time" >> results1''')

for j in range(1,11):
	# for i in ['M','K','N']:
	for i in ['M']:
		# for k in 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
		for x in range(50,801,50):
			# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 4 -f power -f ddr-bw -o power_reports/cake_power_$k-$j -p ./cake_sgemm_test $m;
			if i == 'M':
				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M$k-K5000-N5000-$j ./cake_sgemm_test $k 5000 5000;
				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M$k-K5000-N5000-$j ./mkl_sgemm_test $k 5000 5000;
				os.system("/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M%d-K%d-N%d-%d -p ./cake_sgemm_test %d %d %d" % (x,K,N,j,x,K,N))
				os.system("/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M%d-K%d-N%d-%d -p ./mkl_sgemm_test %d %d %d" % (x,K,N,j,x,K,N))
			elif i == 'K':
				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M5000-K$k-N5000-$j ./cake_sgemm_test 5000 $k 5000;
				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M5000-K$k-N5000-$j ./mkl_sgemm_test 5000 $k 5000;
				os.system("/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M%d-K%d-N%d-%d -p ./cake_sgemm_test %d %d %d" % (M,x,N,j,M,x,N))
				os.system("/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M%d-K%d-N%d-%d -p ./mkl_sgemm_test %d %d %d" % (M,x,N,j,M,x,N))
			else:
				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M5000-K5000-N$k-$j ./cake_sgemm_test 5000 5000 $k; 
				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M5000-K5000-N$k-$j ./mkl_sgemm_test 5000 5000 $k; 
				os.system("/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M%d-K%d-N%d-%d -p ./cake_sgemm_test %d %d %d" % (M,K,x,j,M,K,x))
				os.system("/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M%d-K%d-N%d-%d -p ./mkl_sgemm_test %d %d %d" % (M,K,x,j,M,K,x))



# # run matmul bench for 1..10 threads through intel vtune
# # source /opt/intel/oneapi/setvars.sh;
# cd /home/vnatesh/CAKE_on_CPU/examples;
# # mkdir power_reports;


# # array=( 2 3 4 5 6 8 10 )
# # for i in "${array[@]}"
# for j in {1..10}
# do
# 	for k in 10 50 100 500 1000 1500 2000 5000 3000 3500 4000 4500 5000
# 	do
# 		# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 4 -f power -f ddr-bw -o power_reports/mkl_power_$k-$j -p ./mkl_sgemm_test $m;
# 		perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_power_$k-$j ./mkl_sgemm_test $m;
# 	done
# done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
# scp -r report_stalls vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;

