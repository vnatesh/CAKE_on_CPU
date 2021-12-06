import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys

intel_color = '#0071c5'

# factor of 4 accounts for read/write during packing and unpacking
# def cake_cpu_DRAM_accesses(m,n,k,mc,kc,alpha,p):
# 	return (((float(m*n*k)/(alpha*p*mc) + float(m*n*k)/(p*mc) + 2*m*n) + 4*(m*n) + 2*(m*k + k*n)) / float(10**9))*4	

def cake_cpu_DRAM_accesses(m,n,k,mc,kc,alpha,p):
	return (((float(m*n*k)/(alpha*p*mc) + float(2*m*n*k)/(p*kc) + k*n) + 4*(m*n) + 2*(m*k + k*n)) / float(10**9))*4	


def plot_cake_vs_mkl_cpu(M,N,K,mc,kc,alpha,fname = 'cake_m_vs_mkl', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal', 'CAKE extrapolated', 'MKL extrapolated']
	NUM_CPUs = list(range(1,11))
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[];cake_mem_acc_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; elapsed_time = 0; setting =1;
	#
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			df1 = pandas.read_csv('reports/report_mkl_%d-%d.csv' % (NUM_CPUs[i],j) ,skiprows=17,skipfooter=16)
			df2 = pandas.read_csv('reports/report_mkl_%d-%d.csv' % (NUM_CPUs[i],j),skipfooter=20)
			dram_bw_cpu += (df1['Average']._values[0])
			cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
			gflops_cpu += ((float(M*N*K) / cpu_time) / (10**9))
			#
			df1 = pandas.read_csv('reports/report_cake_sgemm_%d-%d.csv' % (NUM_CPUs[i],j),skiprows=17,skipfooter=16)
			df2 = pandas.read_csv('reports/report_cake_sgemm_%d-%d.csv' % (NUM_CPUs[i],j),skipfooter=20)
			dram_bw_cake += (df1['Average']._values[0])
			cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
			gflops_cake += ((float(M*N*K) / cpu_time) / (10**9))
			if i == 0 and setting == 1:
				elapsed_time += df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
				setting = 0
		#
		dram_bw_cpu_arr.append(dram_bw_cpu / ntrials)
		dram_bw_cake_arr.append(dram_bw_cake / ntrials)
		gflops_cpu_arr.append(gflops_cpu / ntrials)
		gflops_cake_arr.append(gflops_cake / ntrials)
		cake_mem_acc_arr.append(cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,NUM_CPUs[i]) / (elapsed_time/(NUM_CPUs[i]*ntrials)))
		dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; #elapsed_time = 0
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(NUM_CPUs, dram_bw_cpu_arr, label = labels[1],  marker = markers[1], color = intel_color)
	plt.plot(NUM_CPUs, dram_bw_cake_arr, label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(NUM_CPUs, cake_mem_acc_arr, label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	# plt.plot(list(NUM_CPUs), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
	#
	plt.title('(c) Intel DRAM Bandwidth in CAKE M-First vs MKL')
	plt.xlabel("Number of Cores", fontsize = 18)
	# plt.xticks(list(range(1,21,2)))
	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
	plt.legend(loc = "center right", prop={'size': 10})
	# plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[3], color = intel_color)
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[2], color = colors[5])
	#
	# extrapolation lines
	x = np.array(list(range(10,21)))
	# y = [gflops_cpu_arr[-1]]*11
	y = [gflops_cpu_arr[-1] + (gflops_cpu_arr[-1] - gflops_cpu_arr[-2])*i - 0.6*i*i for i in range(0,8)]
	y += 3*[y[-1]]
	plt.plot(x, y,color = intel_color,linestyle = 'dashed', label = labels[5])
	#
	plt.plot(list(range(5,21)), [gflops_cake_arr[4]+i*(gflops_cake_arr[5]-gflops_cake_arr[4]) for i in range(16)], 
		label = labels[4], linewidth = 2, linestyle = 'dashed', color = colors[5])
	plt.xticks(list(range(1,21)))
	#
	plt.title('(d) Computation Throughput in CAKE M-First vs MKL')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_cake_vs_mkl_cpu(23040,23040,23040,144,144,1,ntrials=1)


# if __name__ == '__main__':
# 	plot_cake_vs_mkl_cpu(23040,23040,23040,144,144,1,ntrials=int(sys.argv[1]))