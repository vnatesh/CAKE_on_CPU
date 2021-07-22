import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys


def plot_cake_vs_amd_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_amd', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'OpenBlas Observed','CAKE Optimal', 'CAKE extrapolated', 'OpenBlas extrapolated']
	NUM_CPUs = list(range(1,17))
	#
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0
	#
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			a = open("reports/report_cake_amd_%d-%d.csv" % (NUM_CPUs[i], j) ,'r').read()
			data = [q for q in a.split('\n') if 'PID' in q][0].split(',')
			cpu_time = float(data[1]) / NUM_CPUs[i]
			dram_bw_cake += (((float(data[2])*50000*64) / cpu_time) / (10**9))
			gflops_cake += ((float(M*N*K) / cpu_time) / (10**9))
			#
			a = open("reports/report_openblas_amd_%d-%d.csv" % (NUM_CPUs[i], j),'r').read()
			data = [w for w in a.split('\n') if 'PID' in w][0].split(',')
			cpu_time = float(data[1]) / NUM_CPUs[i]
			dram_bw_cpu += (((float(data[2])*50000*64) / cpu_time) / (10**9))
			gflops_cpu += ((float(M*N*K) / cpu_time) / (10**9))
			# print((((float(data[2])*50000*64) / cpu_time) / (10**9)))
		#
		dram_bw_cpu_arr.append(dram_bw_cpu / ntrials)
		dram_bw_cake_arr.append(dram_bw_cake / ntrials)
		gflops_cpu_arr.append(gflops_cpu / ntrials)
		gflops_cake_arr.append(gflops_cake / ntrials)
		dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0
	#
	plt.figure(figsize = (6,4))
	plt.plot(NUM_CPUs, dram_bw_cpu_arr, label = labels[1],  marker = markers[1], color = colors[3])
	plt.plot(NUM_CPUs, dram_bw_cake_arr, label = labels[0],  marker = markers[0], color = colors[5])
	#
	plt.title('(a) DRAM Bandwidth in CAKE vs OpenBlas')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
	plt.legend(loc = "center right", prop={'size': 10})
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	x = np.array(list(range(15,33)))
	# y = [gflops_cpu_arr[-1]]*11
	y = [gflops_cpu_arr[-2] + (gflops_cpu_arr[-1] - gflops_cpu_arr[-2])*i - 0.6*i*i for i in range(16)]
	y += 2*[y[-1]]
	plt.plot(x, y, color = colors[3], linestyle = 'dashed', label = labels[4])
	#
	plt.plot(list(range(15,33)), [gflops_cake_arr[-2]+i*(gflops_cake_arr[-1]-gflops_cake_arr[-2]) for i in range(18)], 
		label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[5])
	plt.xticks(list(range(2,33,2)))
	#
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[2], color = colors[5])
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('(b) Computation Throughput of CAKE vs OpenBlas')
	plt.xlabel("Number of Cores", fontsize = 18)
	# plt.xticks(NUM_CPUs)
	plt.ylabel("Throughput (GFLOPS/sec)", fontsize = 18)
	plt.legend(loc = "lower right", prop={'size': 12})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


if __name__ == '__main__':
	plot_cake_vs_amd_cpu(23040,23040,23040,144,144,1,ntrials=int(sys.argv[1]))
