#!/bin/bash


# run matmul bench for 1..10 threads through intel vtune
# source /opt/intel/oneapi/setvars.sh;
cd /home/vnatesh/CAKE_on_CPU/examples;
# mkdir power_reports;


# compile mkl_dgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 



# compile cake_sgemm_test
make;

arrVar1=(1 10 50 100)
arrVar2=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000)
arrVar=("${arrVar1[@]}" "${arrVar2[@]}")

# for value in "${arrVar[@]}"
# do
#      echo $value
# done


echo "algo,M,K,N,time" >> results_8N;

for j in {1..10}
do
	# for n in {500..4000..500}
	# do
	# for m in {500..4000..500}
	#for m in {250..2000..250}
	for m in {125..1000..125}
	do
		# for k in 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 500 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
		for k in "${arrVar[@]}"
		# for k in {500..4000..500}
		do
			./mkl_sgemm_test $m $k $m;
			./cake_sgemm_test $m $k $m;
			sleep 1;
		done
	done
	# done
done






# # M=1500;
# # K=500;
# N=500;


# for k in {10..100..10}
# do
# 	echo "algo,M,K,N,time" >> results$k;
# 	for j in {1..10}
# 	do
# 		for i in {1..3}
# 		do
# 			# for k in 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 500 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
# 			for x in {500..1500..500}
# 			do
# 				# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 4 -f power -f ddr-bw -o power_reports/cake_power_$k-$j -p ./cake_sgemm_test $m;
# 				if [[ "$i" -eq 1 ]]
# 				then
# 					# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M$k-K500-N500-$j ./cake_sgemm_test $k 500 500;
# 					# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M$k-K500-N500-$j ./mkl_sgemm_test $k 500 500;
# 					# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports1/cake_socw_M$k-K$K-N$N-$j -p ./cake_sgemm_test $k $K $N;
# 					# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports1/mkl_socw_M$k-K$K-N$N-$j -p ./mkl_sgemm_test $k $K $N;
# 					./cake_sgemm_test $x $k $N;
# 					./mkl_sgemm_test $x $k $N;

# 				# elif [[ "$i" -eq 2 ]]
# 				# then
# 				# 	# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M500-K$k-N500-$j ./cake_sgemm_test 500 $k 500;
# 				# 	# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M500-K$k-N500-$j ./mkl_sgemm_test 500 $k 500;
# 				# 	# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M500-K$k-N500-$j -p ./cake_sgemm_test 500 $k 500;
# 				# 	# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M500-K$k-N500-$j -p ./mkl_sgemm_test 500 $k 500;
# 				# 	./cake_sgemm_test $M $x $N;
# 				# 	./mkl_sgemm_test $M $x $N;
# 				# else
# 				# 	# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M500-K500-N$k-$j ./cake_sgemm_test 500 500 $k; 
# 				# 	# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M500-K500-N$k-$j ./mkl_sgemm_test 500 500 $k; 
# 				# 	/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M500-K500-N$k-$j -p ./cake_sgemm_test 500 500 $k;
# 				# 	/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M500-K500-N$k-$j -p ./mkl_sgemm_test 500 500 $k;
# 				fi
# 			done
# 		done
# 	done
# done




# # array=( 2 3 4 5 6 8 10 )
# # for i in "${array[@]}"
# for j in {1..10}
# do
# 	for k in 10 50 100 500 1000 1500 2000 500 3000 3500 4000 4500 500
# 	do
# 		# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 4 -f power -f ddr-bw -o power_reports/mkl_power_$k-$j -p ./mkl_sgemm_test $m;
# 		perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_power_$k-$j ./mkl_sgemm_test $m;
# 	done
# done

# python python/plots.py
# scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
# scp -r report_stalls vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;

# scp -r power_reports1 vikas@10.0.0.185:/Users/vikas/Documents/test/;


# #!/bin/bash


# # run matmul bench for 1..10 threads through intel vtune
# # source /opt/intel/oneapi/setvars.sh;
# cd /home/vnatesh/CAKE_on_CPU/examples;
# # mkdir power_reports;


# # compile mkl_dgemm_test with Intel MKL
# gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
# -Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
# -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
#  -lpthread -lm -ldl -o mkl_sgemm_test; 

# # compile cake_sgemm_test
# make;
# echo "algo,M,K,N,time" >> results;


# for j in {1..10}
# do
# 	for i in {1..3}
# 	do
# 		# for k in 10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 500 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
# 		for k in {50..500..50}
# 		do
# 			# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 4 -f power -f ddr-bw -o power_reports/cake_power_$k-$j -p ./cake_sgemm_test $m;
# 			if [[ "$i" -eq 1 ]]
# 			then
# 				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M$k-K500-N500-$j ./cake_sgemm_test $k 500 500;
# 				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M$k-K500-N500-$j ./mkl_sgemm_test $k 500 500;
# 				/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M$k-K500-N500-$j -p ./cake_sgemm_test $k 500 500;
# 				/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M$k-K500-N500-$j -p ./mkl_sgemm_test $k 500 500;
# 			elif [[ "$i" -eq 2 ]]
# 			then
# 				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M500-K$k-N500-$j ./cake_sgemm_test 500 $k 500;
# 				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M500-K$k-N500-$j ./mkl_sgemm_test 500 $k 500;
# 				/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M500-K$k-N500-$j -p ./cake_sgemm_test 500 $k 500;
# 				/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M500-K$k-N500-$j -p ./mkl_sgemm_test 500 $k 500;
# 			else
# 				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/cake_M500-K500-N$k-$j ./cake_sgemm_test 500 500 $k; 
# 				# perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_M500-K500-N$k-$j ./mkl_sgemm_test 500 500 $k; 
# 				/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/cake_socw_M500-K500-N$k-$j -p ./cake_sgemm_test 500 500 $k;
# 				/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o power_reports/mkl_socw_M500-K500-N$k-$j -p ./mkl_sgemm_test 500 500 $k;
# 			fi
# 		done
# 	done
# done


# # # array=( 2 3 4 5 6 8 10 )
# # # for i in "${array[@]}"
# # for j in {1..10}
# # do
# # 	for k in 10 50 100 500 1000 1500 2000 500 3000 3500 4000 4500 500
# # 	do
# # 		# /opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 4 -f power -f ddr-bw -o power_reports/mkl_power_$k-$j -p ./mkl_sgemm_test $m;
# # 		perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o power_reports/mkl_power_$k-$j ./mkl_sgemm_test $m;
# # 	done
# # done

# # python python/plots.py
# # scp -r plots vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;
# # scp -r report_stalls vikas@10.0.0.185:/Users/vikas/Documents/cake_pytorch/test/;

