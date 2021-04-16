import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re

intel_color = '#0071c5'

def op_intensity(M,N,K,M_sr,N_sr):
	M,N,K= float(M), float(N), float(K)
	num_macs = M*N*K
	P = M*N
	D = (M/M_sr)*K*N 
	W = (N/N_sr)*M*K 
	return (num_macs / (P+D+W))


def roofline():
	plt.rcParams.update({'font.size': 16})
	s2 = op_intensity(32,64,32,32,32)
	s4 = op_intensity(32,64,32,16,16)
	s8 = op_intensity(32,64,32,8,8)
	# Peak performance (tile mults/cycle)
	P_max = 32
	# Peak DRAM bw (tiles/cycle)
	b_max = 2
	#
	oi_list = [1/8., 1/4., 1/2., 1, 2, 3, 4, 6, 8, 16, 32, 64]
	p = [min(P_max, b_max*x) for x in oi_list]
	plt.plot(oi_list, p,'-b', label  = "roofline", linewidth=4, color = 'black')
	plt.scatter([s2],[0.893], color = 'r', s=40)
	plt.scatter([s4],[1.904], color = 'b', s=40)
	plt.scatter([s8],[3.529], color = 'g', s=40)
	#
	plt.title('Roofline Model for Various Pod Sizes')
	plt.xscale('log', basex=2)
	plt.yscale('log', basey=2)
	plt.xlabel('Operational Intensity (tile mults / tile)', fontsize = 18)
	plt.ylabel('Performance (tile mults / cycle)', fontsize = 18)
	# plt.grid()
	plt.axvline(16, label = 'memory/compute\nboundary', linestyle='dashed')
	# plt.text(16,0,'memory vs compute boundary',rotation=90)
	plt.annotate("s = 2", (3.5 ,0.8))
	plt.annotate("s = 4", (2 ,1.5))
	plt.annotate("s = 8", (1.5 ,2.3))
	plt.legend(loc = "upper left", prop={'size': 15})
	plt.savefig("roofline.pdf", bbox_inches='tight')
	plt.show()
	plt.clf()



def plot_mem_size_R(fname = 'mem_size_R', NUM_SA = 64):
	# 100 linearly spaced numbers
	R = np.linspace(1.01,2,100)
	SZ_sr = (NUM_SA*R) / (R-1)
	plt.figure(figsize=(4,3)) 
	plt.plot(R,SZ_sr, 'r')
	# plt.title("Local memory size as a function of R")
	plt.xlabel("R", fontsize = 18)
	plt.ylabel("memory size (tiles)", fontsize = 18)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.close('all')


def ext_mem_accesses(M,K,N,Sx,Sy,s,alpha,tile_sz):
	'''
		inputs : MMM dims and arch params
		return number accesses (loads and stores) to external memory (DRAM)
	'''
	M_sr = s*((Sx*Sy)/(s*s))*tile_sz
	K_sr = s*tile_sz
	N_sr = s*((Sx*Sy)/(s*s))*alpha*tile_sz
	num_cbs_blks = K/K_sr * M/M_sr * N/N_sr
	dram_transfers_per_blk = (M_sr*K_sr +  K_sr*N_sr) # CAKE transmits only weight and data
	result = M*N
	return(num_cbs_blks*dram_transfers_per_blk + result)


def local_mem_size(Sx,Sy,s,alpha,tile_sz):
	'''
		inputs : MMM dims and arch params
		return : local memory (SRAM) size in bytes
	'''
	M_sr = s*((Sx*Sy)/(s*s))*tile_sz
	K_sr = s*tile_sz
	N_sr = s*((Sx*Sy)/(s*s))*alpha*tile_sz
	weights = M_sr*K_sr
	data = K_sr*N_sr
	partial = M_sr*N_sr
	return((weights+data+partial)*4)


# number of dram accesses for sgemm in bytes...includes copy of input matrices due to packing
def cake_cpu_DRAM_accesses(m,n,k,mc,kc,alpha,p):
	return (((float(m*n*k)/(alpha*p*mc) + float(m*n*k)/(p*mc) + m*n) + 3*(m*n + m*k + k*n)) / float(10**9))*4	

def cake_cpu_cache_sz(mc,kc,alpha,p):
	return (p*mc*kc*(1+alpha) + alpha*(p**2)*(mc**2))*4	

# Goto algorithm doesn't shape the MMM block to adjust 
# for different number of cores p. Instead it shapes block to maximize cache usage
# while leaving approx half of the cache space empty to allow for data reuse and prevent evictions
def mkl_cpu_DRAM_accesses(m,n,k,mc,kc,nc):
	return (((float(m*k*n)/(nc)) + (n*k) + (float(m*n*k)/kc)) / float(10**9))*4	


def plot_cake_vs_mkl_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_mkl', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal']
	NUM_CPUs = list(range(1,11))
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[];cake_mem_acc_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; elapsed_time = 0
	#
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			df1 = pandas.read_csv('reports/report_mkl_%d-%d.csv' % (NUM_CPUs[i],j) ,skiprows=17,skipfooter=17)
			df2 = pandas.read_csv('reports/report_mkl_%d-%d.csv' % (NUM_CPUs[i],j),skipfooter=20)
			dram_bw_cpu += (df1['Average']._values[0])
			cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
			gflops_cpu += ((float(M*N*K) / cpu_time) / (10**9))
			#
			df1 = pandas.read_csv('reports/report_cake_dgemm_%d-%d.csv' % (NUM_CPUs[i],j),skiprows=17,skipfooter=17)
			df2 = pandas.read_csv('reports/report_cake_dgemm_%d-%d.csv' % (NUM_CPUs[i],j),skipfooter=20)
			dram_bw_cake += (df1['Average']._values[0])
			cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
			gflops_cake += ((float(M*N*K) / cpu_time) / (10**9))
			elapsed_time += df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
		#
		dram_bw_cpu_arr.append(dram_bw_cpu / ntrials)
		dram_bw_cake_arr.append(dram_bw_cake / ntrials)
		gflops_cpu_arr.append(gflops_cpu / ntrials)
		gflops_cake_arr.append(gflops_cake / ntrials)
		cake_mem_acc_arr.append(cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,NUM_CPUs[i]) / (elapsed_time/ntrials))
		dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; elapsed_time = 0
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(NUM_CPUs, dram_bw_cpu_arr, label = labels[1],  marker = markers[1], color = intel_color)
	plt.plot(NUM_CPUs, dram_bw_cake_arr, label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(NUM_CPUs, cake_mem_acc_arr, label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	# plt.plot(list(NUM_CPUs), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
	#
	plt.title('(a) Intel DRAM Bandwidth in CAKE vs MKL')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
	plt.legend(loc = "center right", prop={'size': 10})
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')
	#
	# plt.subplot(1, 2, 2)
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[3], color = intel_color)
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[2], color = colors[5])
	#
	plt.title('(b) Computation Throughput in CAKE vs MKL')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	# plt.suptitle('Performance of CAKE vs MKL', fontsize = 18)
	plt.show()
	plt.clf()
	plt.close('all')



def plot_cake_vs_armpl_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_armpl', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'ARMPL Observed', 'CAKE Optimal']
	NUM_CPUs = [1,2,3,4]
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[];cake_mem_acc_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; cake_mem_acc = 0
	#
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			a = open('reports_arm/report_arm_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
			dram_bw_cpu += ((int(re.search(r'\d+', a[5]).group())*64.0) / cpu_time) / (10.0**9)
			gflops_cpu += (float(M*N*K) / cpu_time) / (10**9)
			#
			a = open('reports_arm/report_cake_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
			dram_bw_cake += ((int(re.search(r'\d+', a[5]).group())*64.0) / cpu_time) / (10.0**9)
			gflops_cake += (float(M*N*K) / cpu_time) / (10**9)# / (float(NUM_CPUs[i]))
			cake_mem_acc += cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,NUM_CPUs[i]) / cpu_time
		#
		dram_bw_cpu_arr.append(dram_bw_cpu / ntrials)
		dram_bw_cake_arr.append(dram_bw_cake / ntrials)
		gflops_cpu_arr.append(gflops_cpu / ntrials)
		gflops_cake_arr.append(gflops_cake / ntrials)
		cake_mem_acc_arr.append(cake_mem_acc / ntrials)
		dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; cake_mem_acc = 0
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(dram_bw_cpu_arr), label = labels[1],  marker = markers[1], color = colors[4])
	plt.plot(list(NUM_CPUs), list(dram_bw_cake_arr), label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(list(NUM_CPUs), list(cake_mem_acc_arr), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	plt.title('(a) DRAM Bandwidth in CAKE vs ARMPL')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
	plt.legend(loc = "center right", prop={'size': 10})
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[2], color = colors[5])
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[3], color = colors[4])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(b) Computation Throughput of CAKE vs ARMPL')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	# plt.suptitle('Performance of CAKE vs ARMPL', fontsize = 18)
	plt.show()
	plt.clf()
	plt.close('all')



def plot_cake_vs_amd_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_amd', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'OpenBlas Observed','CAKE Optimal']
	NUM_CPUs = list(range(1,17))
	#
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0
	#
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			a = open("reports/report_cake_amd_%d-%d.csv" % (NUM_CPUs[i], j) ,'r').read()
			data = [i for i in a.split('\n') if 'PID' in i][0].split(',')
			cpu_time = float(data[1]) / NUM_CPUs[i]
			dram_bw_cake += (((float(data[2])*50000*64) / cpu_time) / (10**9))
			gflops_cake += ((float(M*N*K) / cpu_time) / (10**9))
			#
			a = open("reports/report_openblas_amd_%d-%d.csv" % (NUM_CPUs[i], j),'r').read()
			data = [i for i in a.split('\n') if 'PID' in i][0].split(',')
			cpu_time = float(data[1]) / NUM_CPUs[i]
			dram_bw_cpu += (((float(data[2])*50000*64) / cpu_time) / (10**9))
			gflops_cpu += ((float(M*N*K) / cpu_time) / (10**9))
			print((((float(data[2])*50000*64) / cpu_time) / (10**9)))
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
	# plt.show()
	# plt.clf()
	# plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[2], color = colors[5])
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('(b) Computation Throughput of CAKE vs OpenBlas')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Throughput (GFLOPS/sec)", fontsize = 18)
	plt.legend(loc = "lower right", prop={'size': 12})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')


def get_LLC_pmbw(exp, arr_size, file_name, ncores):
	a = open(file_name,'r').read().split('\n')
	a = [i for i in a if exp in i]
	a = [i.split('\t') for i in a]
	a = [i for i in a if i[6] == 'areasize=%d' % arr_size]
	int_bw = []
	NUM_CPUs = range(1,ncores+1)
	for i in range(len(NUM_CPUs)):
		int_bw.append(float(a[i][-2][10:]) / (10**9))
	#
	return int_bw


def plot_internal_bw_cpu(cpu):
	labels = ['Intel i9','ARM Cortex v8 A53', 'AMD Ryzen 9 5950X']
	plt.rcParams.update({'font.size': 12})
	plt.figure(figsize = (6,4))
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	int_bw_intel = get_LLC_pmbw('ScanWrite64PtrUnrollLoop', 2**24, 'stats.txt', 10)
	int_bw_arm = get_LLC_pmbw('ScanWrite64PtrUnrollLoop', 2**18, 'stats_arm.txt', 4)
	int_bw_amd = get_LLC_pmbw('ScanWrite64PtrUnrollLoop', 2**24, 'stats_amd.txt', 16)
	#
	if 'Intel' in cpu:
		NUM_CPUs = list(range(1,11))
		plt.plot(NUM_CPUs[:10], int_bw_intel, label = labels[0], marker = markers[0], color = colors[1])
	elif 'ARM' in cpu:
		NUM_CPUs = list(range(1,5))
		plt.plot(NUM_CPUs[:4], int_bw_arm, label = labels[1], marker = markers[1], color = colors[1])
	elif'AMD' in cpu:
		NUM_CPUs = list(range(1,17))
		plt.plot(NUM_CPUs, int_bw_amd, label = labels[2], marker = markers[2], color = colors[1])
	#
	plt.title('(c) Internal Bandwidth On %s CPU' % cpu)
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Bandwidth (GB/s)", fontsize = 18)
	# plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_cache_bw.pdf" % cpu, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')



def plot_cake_vs_mkl_sparse(fname = 'cake_vs_mkl_sparse'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal']
	density = [(1-i)*100 for i in [0.5,0.4,0.3,0.2,0.1,0.05,0.02,0.01,0.005]]
	N = 5000
	K = 2048
	M = 33708
	#
	df1 = pandas.read_csv('sparse.csv')
	cpu_times = df1['cpu_time']._values
	dram_bw_cpu = df1['bw']._values  # number of GB transferred b/w processors and DRAM
	gflops_cpu = [(float(M*N*K) / cpu_times[i]) / (10**9) for i in range(len(density))] # / (float(density[i]))
	#
	#
	dram_bw_cake = [7.284 for i in range(len(density))]
	gflops_cake = [(float(M*N*K) / 0.607994) / (10**9) for i in range(len(density))]
	dram_bw_cake_theo = [cake_cpu_DRAM_accesses(M,N,K,144,144,1,10) for i in density]
	#
	plt.plot(list(density), list(dram_bw_cpu), label = labels[1],  marker = markers[1], color = intel_color)
	# plt.plot(list(density), list(dram_bw_cake_theo), label = labels[2],  marker = markers[3], color = colors[5], linestyle = 'dashed')
	plt.plot(list(density), list(dram_bw_cake), label = labels[0],  marker = markers[0], color = colors[5])
	#
	plt.title('(a) DRAM Bandwidth for SparseMM in CAKE vs MKL')
	plt.xlabel("Percent Sparsity", fontsize = 18)
	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
	plt.legend(loc = "center left", prop={'size': 14})
	# plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.plot(list(density), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[5])
	plt.plot(list(density), list(gflops_cpu), label = labels[1],  marker = markers[3], color = intel_color)
	#
	plt.title('(b) Comp. Throughput of SparseMM in CAKE vs MKL')
	plt.xlabel("Percent Sparsity", fontsize = 18)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 14})
	# plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	


def plot_bank_area():
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# labels = ['']	
	NUM_BANKs = [8,16,32,64]
	area = [1,1.052,1.155,1.363]
	plt.plot(list(NUM_BANKs), area,  marker = markers[2], color = colors[1])
	plt.title('Normalized Area Cost of Adding DRAM Banks')
	plt.xlabel("Number of Banks", fontsize = 12)
	plt.ylabel("Area Cost", fontsize = 12)
	plt.xticks(NUM_BANKs)
	# plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



if __name__ == '__main__':
	# plot_cake_vs_mkl_cpu(23040,23040,23040,144,144,1)
	# plot_cake_vs_armpl_cpu(3000,3000,3000,48,48,1)
	# plot_cake_vs_amd_cpu(23040,23040,23040,144,144,1)
	# plot_internal_bw_cpu('Intel')
	# plot_internal_bw_cpu('AMD')
	# plot_internal_bw_cpu('ARM')
	# plot_cake_vs_mkl_sparse()
	# plot_cake_vs_mkl_transformer(10,144,144,1)









#------------ OLD PLOTS--------------#


# labels = list(map(str,range(1000,10001,1000)))
# women_means = [0.450558,0.381210,0.354841,0.347499,0.341100,0.334189,0.336083,0.328989,0.326314,0.316413]
# men_means = [0.549442,0.618790,0.645159,0.652501,0.658900,0.665811,0.663917,0.671011,0.673686,0.683587]
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='GEMM')
# rects2 = ax.bar(x + width/2, women_means, width, label='Packing')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('percent of runtime')
# ax.set_title('Packing Cost in CAKE')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# fig.tight_layout()
# plt.show()



# def plot_cake_vs_mkl_transformer(p,mc,kc,alpha,fname = 'cake_vs_mkl_weight'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['MKL Observed', 'CAKE Observed','CAKE Optimal', 'MKL Optimal']
# 	batch_size = list(range(1000,10001,1000))
# 	gflops_cpu=[];gflops_cake=[];dram_bw_cake=[];dram_bw_cpu=[];cake_mem_acc=[]
# 	K = 8000
# 	M = 33708
# 	#
# 	for i in range(len(batch_size)):
# 		df1 = pandas.read_csv('reports/report_mkl_%d.csv' % batch_size[i],skiprows=17,skipfooter=17)
# 		df2 = pandas.read_csv('reports/report_mkl_%d.csv' % batch_size[i],skipfooter=25)
# 		dram_bw_cpu.append(df1['Average']._values[0])
# 		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(p)
# 		gflops_cpu.append((float(M*K*batch_size[i]) / cpu_time) / (10**9)) # / (float(batch_size[i]))
# 	#
# 		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % batch_size[i],skiprows=17,skipfooter=17)
# 		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % batch_size[i],skipfooter=25)
# 		dram_bw_cake.append(df1['Average']._values[0])
# 		elapsed_time = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
# 		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(p)
# 		gflops_cake.append((float(M*K*batch_size[i]) / cpu_time) / (10**9)) # / (float(batch_size[i]))
# 		cake_mem_acc.append(cake_cpu_DRAM_accesses(M,batch_size[i],K,mc,kc,alpha,p) / elapsed_time)
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(list(batch_size), list(dram_bw_cpu), label = labels[0],  marker = markers[0], color = intel_color)
# 	plt.plot(list(batch_size), list(dram_bw_cake), label = labels[1],  marker = markers[1], color = colors[5])
# 	plt.plot(list(batch_size), list(cake_mem_acc), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
# 	#
# 	plt.title('(a) DRAM Bandwidth in Gradient Computation')
# 	plt.xlabel("Batch Size", fontsize = 16)
# 	plt.xticks(batch_size)
# 	plt.ylabel("DRAM Bw (GB/s)", fontsize = 16)
# 	plt.legend(loc = "center right", prop={'size': 10})
# 	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')
# 	#
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(list(batch_size), list(gflops_cpu), label = labels[0],  marker = markers[3], color = intel_color)
# 	plt.plot(list(batch_size), list(gflops_cake), label = labels[1],  marker = markers[2], color = colors[5])
# 	#
# 	plt.title('(b) Gradient Computation in CAKE vs MKL')
# 	plt.xlabel("Batch Size", fontsize = 16)
# 	plt.xticks(batch_size)
# 	plt.ylabel("Comp. Throughput (GFLOP/s)", fontsize = 16)
# 	plt.legend(loc = "lower right", prop={'size': 12})
# 	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')
# 	#


# def op_intensity_cpu(m,n,k,mc,kc,alpha,p):
# 	return ((m*n*k) / (cake_cpu_DRAM_accesses(m,n,k,mc,kc,alpha,p) * (10**9)))

# def roofline_cpu(m,n,k,p,cpu):
# 	plt.rcParams.update({'font.size': 16})
# 	colors = ['b','g','aqua','k','m','r']
# 	if 'Intel' in cpu:	
# 		df1 = pandas.read_csv('reports/report_mkl_%d.csv' % p,skiprows=17,skipfooter=17)
# 		df2 = pandas.read_csv('reports/report_mkl_%d.csv' % p,skipfooter=20)
# 		dram_bw_cpu = df1['Average']._values[0]
# 		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(p)
# 		elapsed_time = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
# 		oi_cpu = float(2*m*n*k) / (elapsed_time * dram_bw_cpu * 10**9)
# 		gflops_cpu = (float(2*m*n*k) / cpu_time) / (10**9)
# 		#
# 		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % p,skiprows=17,skipfooter=17)
# 		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % p,skipfooter=20)
# 		dram_bw_cake= df1['Average']._values[0]
# 		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(p)
# 		elapsed_time = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
# 		oi_cake = float(2*m*n*k) / (elapsed_time * dram_bw_cake * 10**9)
# 		gflops_cake = (float(2*m*n*k) / cpu_time) / (10**9)
# 		#
# 		color = intel_color
# 		# Peak performance (GFLOP/s https://gadgetversus.com/processor/intel-core-i9-10900k-specs/ 
# 		# 5.3 clock rate * 2 FMAs/clock * (AVX2 256bit reg / 32 bit float) * 10 cores
# 		P_max = 848*2
# 		# Peak DRAM bw (GB/s)
# 		b_max = 40
# 		#
# 	# elif 'AMD' in cpu:
# 	# 	a = open("reports/report_cake_amd_%d.csv" % p,'r').read()
# 	# 	data = [i for i in a.split('\n') if 'PID' in i][0].split(',')
# 	# 	cpu_time = float(data[1]) / p
# 	# 	dram_bw_cake = ((float(data[2])*50000*64) / cpu_time) / (10**9)
# 	# 	oi_cake = float(2*m*n*k) / (cpu_time * dram_bw_cake * 10**9)
# 	# 	gflops_cake = (float(2*m*n*k) / cpu_time) / (10**9)
# 	# 	#
# 	# 	a = open("reports/report_openblas_amd_%d.csv" % p,'r').read()
# 	# 	data = [i for i in a.split('\n') if 'PID' in i][0].split(',')
# 	# 	cpu_time = float(data[1]) / p
# 	# 	dram_bw_cpu = ((float(data[2])*50000*64) / cpu_time) / (10**9)
# 	# 	oi_cpu = float(2*m*n*k) / (cpu_time * dram_bw_cpu * 10**9)
# 	# 	gflops_cpu = (float(2*m*n*k) / cpu_time) / (10**9)
# 	# 	#
# 	# 	color = colors[3]
# 	# 	# Peak performance (GFLOP/s https://gadgetversus.com/processor/intel-core-i9-10900k-specs/ 
# 	# 	# 4.9 max clock rate * 2 FMAs/clock * (AVX2 256bit reg / 32 bit float) * 16 cores
# 	# 	P_max = 1254*2
# 	# 	# Peak DRAM bw (GB/s)
# 	# 	b_max = 47
# 	#
# 	oi_list = list(range(0,201))
# 	p = [min(P_max, b_max*x) for x in oi_list]
# 	plt.figure(figsize = (6,4))
# 	plt.plot(oi_list, p,'-b', label  = "roofline", linewidth=4, color = 'black')
# 	plt.scatter([oi_cpu],[gflops_cpu], color = intel_color, s=40)
# 	plt.scatter([oi_cake],[gflops_cake], color = 'r', s=40)
# 	#
# 	plt.title('Roofline Model %s CPU' % cpu)
# 	# plt.xscale('log', basex=2)
# 	# plt.yscale('log', basey=2)
# 	plt.xlabel('Operational Intensity (FLOP/byte)', fontsize = 16)
# 	plt.ylabel('Comp. Throughput (GFLOP/s)', fontsize = 16)
# 	# plt.grid()
# 	plt.axvline(P_max/b_max, label = 'memory/compute\nboundary', linestyle='dashed')
# 	# plt.text(16,0,'memory vs compute boundary',rotation=90)
# 	plt.annotate("MKL", (oi_cpu ,gflops_cpu-150))
# 	plt.annotate("CAKE", (oi_cake ,gflops_cake-150))
# 	# plt.annotate("s = 8", (1.5 ,2.3))
# 	plt.legend(loc = "lower right", prop={'size': 15})
# 	# plt.savefig("roofline.pdf", bbox_inches='tight')
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')

