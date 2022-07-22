import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re

intel_color = '#0071c5'
os.chdir("/Users/vikas/Documents/test")

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
	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal', 'CAKE extrapolated', 'MKL extrapolated']
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
	print(gflops_cpu_arr)
	plt.figure(figsize = (6,4))
	plt.plot(NUM_CPUs, dram_bw_cpu_arr, label = labels[1],  marker = markers[1], color = intel_color)
	plt.plot(NUM_CPUs, dram_bw_cake_arr, label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(NUM_CPUs, cake_mem_acc_arr, label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	# plt.plot(list(NUM_CPUs), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
	#
	plt.title('(a) Intel DRAM Bandwidth in CAKE vs MKL')
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
	plt.title('(b) Computation Throughput in CAKE vs MKL')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



def plot_cake_vs_armpl_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_armpl', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'ARMPL Observed', 'CAKE Optimal','CAKE extrapolated', 'ARMPL extrapolated']
	NUM_CPUs = [1,2,3,4]
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[];cake_mem_acc_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; cake_mem_acc = 0
	#
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			a = open('reports_arm/report_arm_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
			# multiply by 64 bytes since external memory reqeust PMU
			# in ARM is expressed in terms of number of cache lines
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
	# plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	x = np.array(list(range(3,9)))
	# y = [gflops_cpu_arr[-1]]*11
	y = [gflops_cpu_arr[-2] + (gflops_cpu_arr[-1] - gflops_cpu_arr[-2])*i - 0.006*i*i for i in range(4)]
	y += 2*[y[-1]]
	plt.plot(x, y, color = colors[4], linestyle = 'dashed', label = labels[4])
	#
	plt.plot(list(range(1,9)), [gflops_cake_arr[0]+i*(gflops_cake_arr[1]-gflops_cake_arr[0]) for i in range(8)], 
		label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[5])
	plt.xticks(list(range(1,9)))
	#
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[2], color = colors[5])
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[3], color = colors[4])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(b) Computation Throughput of CAKE vs ARMPL')
	plt.xlabel("Number of Cores", fontsize = 18)
	# plt.xticks(NUM_CPUs)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	# plt.suptitle('Performance of CAKE vs ARMPL', fontsize = 18)
	plt.show()
	plt.clf()
	plt.close('all')



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
	labels = ['Intel i9 actual','ARM Cortex v8 A53', 'AMD Ryzen 9 5950X', 'extrapolated']
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
		plt.plot(list(range(5,21)), [int_bw_intel[4]+i*(int_bw_intel[5]-int_bw_intel[4]) for i in range(16)], 
			label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[1])
		plt.xticks(list(range(1,21)))
	elif 'ARM' in cpu:
		NUM_CPUs = list(range(1,5))
		plt.plot(NUM_CPUs[:4], int_bw_arm, label = labels[1], marker = markers[1], color = colors[1])
		plt.xticks(NUM_CPUs)
		plt.plot(list(range(1,9)), [int_bw_arm[0]+i*(int_bw_arm[1]-int_bw_arm[0]) for i in range(8)], 
			label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[1])
		plt.xticks(list(range(1,9)))
	elif'AMD' in cpu:
		NUM_CPUs = list(range(1,17))
		plt.plot(NUM_CPUs, int_bw_amd, label = labels[2], marker = markers[2], color = colors[1])
		plt.xticks(NUM_CPUs)
		plt.plot(list(range(13,33)), [int_bw_amd[0]+i*((int_bw_amd[15]-int_bw_amd[13])/2) for i in range(13,33)], 
			label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[1])
		plt.xticks(list(range(2,33,2)))
	#
	plt.title('(c) Internal Bandwidth On %s CPU' % cpu)
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.ylabel("Bandwidth (GB/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_cache_bw.pdf" % cpu, bbox_inches='tight')
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
	plot_cake_vs_mkl_cpu(23040,23040,23040,144,144,1)
	# plot_cake_vs_armpl_cpu(3000,3000,3000,48,48,1)
	# plot_cake_vs_amd_cpu(23040,23040,23040,144,144,1)
	# plot_internal_bw_cpu('Intel')
	# plot_internal_bw_cpu('AMD')
	# plot_internal_bw_cpu('ARM')
	# plot_cake_vs_mkl_sparse()
	# plot_cake_vs_mkl_transformer(10,144,144,1)


#-----------------------NEW PLOTS-------------------------------



def plot_cake_vs_mkl_shape(fname = 'cake_vs_mkl_shape', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal']
	NUM_CPUs = list(range(1,11))
	#
	plt.figure(figsize = (6,4))
	df1 = pandas.read_csv('results_sq')
	for j in range(1000,3001,1000):
		single_core_mkl = df1[(df1['algo'] == 'mkl') & (df1['size'] == j) & (df1['p'] == 1)]['time'].mean()
		single_core_cake = df1[(df1['algo'] == 'cake') & (df1['size'] == j) & (df1['p'] == 1)]['time'].mean()
		speedup_mkl = []
		speedup_cake = []
		for p in NUM_CPUs:
			a = df1[(df1['algo'] == 'mkl') & (df1['size'] == j) & (df1['p'] == p)]['time'].mean()
			speedup_mkl.append(single_core_mkl / a)
			a = df1[(df1['algo'] == 'cake') & (df1['size'] == j) & (df1['p'] == p)]['time'].mean()
			speedup_cake.append(single_core_cake / a)
		#
		plt.plot(NUM_CPUs, speedup_mkl, label = "%d (mkl)" % j, color = colors[(j/1000) - 1],linestyle='dashed',)
		plt.plot(NUM_CPUs, speedup_cake, label = "%d (cake)" % j, color = colors[(j/1000) - 1])
	#
	plt.title('(a) Speedup For Square Matrices in CAKE vs MKL')
	plt.xlabel("Number of Cores (M=N=K)", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Speedup", fontsize = 18)
	plt.legend(title="M=N=K", loc = "upper left", prop={'size': 10})
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

plot_cake_vs_mkl_shape()



def plot_cake_vs_armpl_shape(fname = 'cake_vs_armpl_shape', ntrials=5):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','k','m','r']
	labels = ['CAKE Observed', 'armpl Observed','CAKE Optimal', 'armpl Optimal']
	NUM_CPUs = list(range(1,5))
	#
	plt.figure(figsize = (6,4))
	df1 = pandas.read_csv('results_arm_new')
	for j in range(1000,3001,1000):
		single_core_armpl = df1[(df1['algo'] == 'armpl') & (df1['size'] == j) & (df1['p'] == 1)]['time'].mean()
		single_core_cake = df1[(df1['algo'] == 'cake') & (df1['size'] == j) & (df1['p'] == 1)]['time'].mean()
		speedup_armpl = []
		speedup_cake = []
		for p in NUM_CPUs:
			a = df1[(df1['algo'] == 'armpl') & (df1['size'] == j) & (df1['p'] == p)]['time'].mean()
			q = df1[(df1['algo'] == 'armpl') & (df1['size'] == j) & (df1['p'] == p)]['time'].std()
			speedup_armpl.append(single_core_armpl / a)
			a = df1[(df1['algo'] == 'cake') & (df1['size'] == j) & (df1['p'] == p)]['time'].mean()
			q = df1[(df1['algo'] == 'cake') & (df1['size'] == j) & (df1['p'] == p)]['time'].std()
			speedup_cake.append(single_core_cake / a)
		#
		plt.plot(NUM_CPUs, speedup_armpl, label = "%d (armpl)" % j, color = colors[int(j/1000) - 1],linestyle='dashed',)
		plt.plot(NUM_CPUs, speedup_cake, label = "%d (cake)" % j, color = colors[int(j/1000) - 1])
	#
	plt.title('(b) Speedup For Square Matrices in CAKE vs ARMPL')
	plt.xlabel("Number of Cores (M=N=K)", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Speedup", fontsize = 18)
	plt.legend(title="M=N=K", loc = "upper left", prop={'size': 10}, fontsize = 16)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

plot_cake_vs_armpl_shape()



# intel freq = 4.6 GhZ
# # CAKE
# L1 Data cache refill - 58036091 * 64 = 3714309824
# L1 Data cache access - 8388617968 							L1 hits = 8388617968 - 58036091 = 8330581877
# L2 Data cache refill - 28624629 * 64 = 1831976256
# L2 Data cache access - 284476078							L2 hits = 255851449   
# External Memory Request - 164691807 * 64 = 10540275648		DRAM reqs = 164691807

# # ARMPL
# L1 Data cache refill - 149707795 
# L1 Data cache access - 3820427325 			L1 hits = 3820427325 - 149707795 = 3670719530
# L2 Data cache refill - 227948309 
# L2 Data cache access - 747526490			L2 hits = 747526490 - 227948309 = 519578181
# External Memory Request - 401276879 		DRAM reqs = 401276879
def mem_req_barplot():
	plt.figure(figsize = (6,4))
	N = 4
	clocks_cake = (4.6*10**9) * (3.283)
	clocks_mkl = (4.6*10**9) * (3.243)
	# cake = (3.4, 3.1, 2.5, 1.2)
	# mkl = (0, 0.5, 0.6, 5.1)
	cake = tuple([i*clocks_cake / (100 * 1e9) for i in (2.5, 1.9, 0.9, 0.5)])
	mkl = tuple([i*clocks_mkl / (100 * 1e9) for i in (1.1, 0.4, 1.0, 3.0)])
	# cake = (3.8, 1.3, 2, 2.1)
	# mkl = (4.2, 0.3, 2.1, 2.2)
	labels = ['L1', 'L2', 'L3', 'Main Memory']
	ind = np.arange(N) 
	width = 0.35       
	plt.bar(ind, cake, width, label='Cake', color = "#ee0000")
	plt.bar(ind + width, mkl, width,label='MKL', color = "#0071C5")
	plt.ylabel("Time (billions of clockticks)", fontsize = 16)
	plt.title('(a) Memory Request Stalls on Intel i9', fontsize = 18)
	plt.xticks(ind + width / 2, tuple(labels), fontsize = 14)
	plt.yticks(fontsize = 12)
	plt.legend(loc='best', fontsize = 14)
	plt.savefig("cake_vs_mkl_memreq.pdf", bbox_inches='tight')
	plt.show()
	#
	plt.figure(figsize = (6,4))
	N = 3
	# cake = (8330581877, 255851449, 164691807)
	# armpl = (3670719530, 519578181, 401276879)
	cake = tuple([i / 1e9 for i in (8330581877, 255851449, 164691807)])
	armpl = tuple([i / 1e9 for i in (3670719530, 519578181, 401276879)])
	labels = ['L1 Hits', 'L2 Hits', 'DRAM Requests']
	ind = np.arange(N) 
	width = 0.35       
	plt.bar(ind, cake, width, label='Cake', color = "#ee0000")
	plt.bar(ind + width, armpl, width,label='ARMPL', color = "#0071C5")
	plt.ylabel("Billions of Accesses", fontsize = 16)
	plt.title('(b) Cache and DRAM Accesses on ARM', fontsize = 18)
	plt.xticks(ind + width / 2, tuple(labels), fontsize = 14)
	plt.yticks(fontsize = 12)
	plt.legend(loc='best', fontsize = 14)
	plt.savefig("cake_vs_armpl_memreq.pdf", bbox_inches='tight')
	plt.show()




def plot_small_mat(x, ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['#FFC0CB', '#FA8072', 'r', '#8C000F']
	# K = [10,50,100] + [i*500 for i in range(1,21)]
	M = [i*500 for i in range(2,17,2)]
	K= [1, 10, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
	vals = [1.0,1.25,1.5,2.0]
	thresh = [[] for i in range(len(vals))] 
	#
	for m in M:
		# df1 = pandas.read_csv('results_full') # use this file for N and 2N plots
		df1 = pandas.read_csv('results_%dN' % x)
		diffs = []
		for k in K:
			a = df1[(df1['algo'] == 'cake') & (df1['M'] == m) & (df1['K'] == k) & (df1['N'] == m/x)]['time']
			lat_cake = a.mean()  
			b = df1[(df1['algo'] == 'mkl') & (df1['M'] == m) & (df1['K'] == k) & (df1['N'] == m/x)]['time']
			lat_mkl = b.mean()
			diffs.append(lat_mkl / lat_cake) 
		#
		print(diffs)
		print
		for v in range(len(vals)):
			thresh[v].append(8000)
			# for i in range(len(diffs)-1,-1,-1):
			# 	if diffs[i] >= (1-eps) and diffs[i] <= 1+eps:
			for i in range(len(diffs)):
				if diffs[i] <= vals[v]:
					thresh[v][-1] = K[i]
					print(K[i])
					break
	#
	#
	plt.figure(figsize = (6,4))
	for i in range(len(vals)):
		plt.plot(M, thresh[i], label = '%.2fx threshhold' % vals[i], color = colors[i])
	#
	plt.xticks(list(range(0,8001,1000)))
	plt.yticks(list(range(0,8001,1000)))
	plt.ylim(0,8000)
	# plt.arrow(4000,4000, -1500, -1500, head_width = 20, head_length = 20, width = 0.05)
	# plt.annotate(s='', xy=(4000,4000), xytext=(2500,2500), arrowprops=dict(arrowstyle='<-'))
	# plt.annotate(s='', xy=(3500,3000), xytext=(4500,4000), arrowprops=dict(arrowstyle='<-'))
	# plt.annotate("Increasing\nPerformance", (3000 ,4500))
	# plt.annotate("Decreasing Performance", (4000 ,5000))
	# plt.text(50, 300, "K=%d, N=%d" % (x,y), fontsize = 12)
	plt.title('(a) Throughput Profile For Different Matrix Dimensions')
	plt.xlabel('M = N = x' , fontsize = 18)
	plt.ylabel("K", fontsize = 18)
	plt.legend(loc = "upper right", prop={'size': 12})
	plt.fill_between(M, 0, thresh[3], color = 'magenta')
	plt.fill_between(M, thresh[3], thresh[2], color = 'deeppink')
	plt.fill_between(M, thresh[2], thresh[1], color = 'hotpink')
	plt.fill_between(M, thresh[1], thresh[0], color = 'lightpink')
	plt.savefig("contour_%dN.pdf" % x, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#

# plot_small_mat(1)
# plot_small_mat(2)
plot_small_mat(4)



def plot_cpu_mem_stalls(fname = 'cpu_mem_stalls', ntrials=2):
	width = 0.35       
	num_cat = 4
	labels = ['L1', 'L2', 'L3', 'DRAM']
	ind = np.arange(num_cat) 
	# M = [10, 50, 100, 1000, 2000, 3000,5000,10000]
	M = [10, 50, 100, 1000, 2000, 3000]
	# M = [2000, 3000]
	K = M
	fig, axs = plt.subplots(len(M), len(K),sharex=True)
	fig.suptitle('Percent of Clockticks Stalled On Memory Requests During MM')
	plt.rc('xtick',labelsize=8)
	plt.rc('ytick',labelsize=8)
	# plt.setp(axs, xticks=ind + width / 2, xticklabels=labels)
	for m in range(len(M)):
		for k in range(len(K)):
			cake = [0.0]*len(labels)
			mkl = [0.0]*len(labels)
			for j in range(1,ntrials+1):
				df1 = pandas.read_csv('report_stalls/report_cake_sgemm_%d-%d-%d.csv' % (M[m],K[k],j),skipfooter=25)
				df2 = pandas.read_csv('report_stalls/report_mkl_%d-%d-%d.csv' % (M[m],K[k],j),skipfooter=25)
				for i in range(len(labels)):
					cake[i] += df1[df1['Metric Name'] == str(labels[i] + ' Bound')]['Metric Value']._values[0] 
					mkl[i] += df2[df2['Metric Name'] == str(labels[i] + ' Bound')]['Metric Value']._values[0]
			#
			cake = [i/ntrials for i in cake]
			mkl = [i/ntrials for i in mkl]
			axs[m, k].bar(ind, cake, width, label='Cake')
			axs[m, k].bar(ind + width, mkl, width,label='MKL')
			# axs[m, k].set(ylabel = "Time (% of total clockticks)")
			axs[m, k].set_title('M=N=%d, K=%d' % (M[m],K[k]), fontsize = 8)
			if m==(len(M) - 1):
				plt.setp(axs[m, k], xticks=ind + width / 2, xticklabels=labels)
				axs[m, k].set_xticks(ind + width / 2, tuple(labels))
			# axs[m, k].legend(loc='best')
	#
	plt.legend(loc='best')
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cpu_mem_stalls()


# where the aspect ratio is the ratio of the size of K relative to M or N.
def plot_local_mem_sz(fname = 'local_mem_sz'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['g','b','aqua','k','m','r']
	labels = ['actual', 'extrapolated']
	NUM_CPUs = range(1,21)
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs)[:10], [2*cake_cpu_cache_sz(144,144,1,i)/1e6 for i in list(NUM_CPUs)[:10]],
		marker = markers[1], label = labels[0], color = colors[0]) # need 2x the cache size to minimize evictions
	plt.plot(list(NUM_CPUs)[9:], [2*cake_cpu_cache_sz(144,144,1,i)/1e6 for i in list(NUM_CPUs)[9:]],
		linestyle='dashed', label = labels[1], linewidth = 2, color = colors[0]) # need 2x the cache size to minimize evictions
	plt.title("Local Memory Size vs. Compute Power",fontsize = 18)
	plt.ticklabel_format(style = 'plain')
	plt.xticks(NUM_CPUs)
	plt.xlabel("Number of Cores",fontsize = 18)
	plt.xlim(xmin=0)
	plt.ylabel("Local Memory Size (MB)",fontsize = 18)
	plt.ylim(ymin=0)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_local_mem_sz()



import math

# def cb_shape(p,alpha,L):
# 	mc = int(math.sqrt(float(L/4.0) / (2*(p + alpha*p) + alpha*p*p)))
# 	m = p*mc
# 	k = mc
# 	n = alpha*p*mc
# 	return (m, k, n)

# def num_cb_blks(M,K,N,p,alpha,L):
# 	M_sr, K_sr, N_sr = cb_shape(p,alpha,L)
# 	return(K/K_sr * M/M_sr * N/N_sr)

# def t_comp(p,alpha,L,mr,nr):
# 	M_sr, K_sr, N_sr = cb_shape(p,alpha,L)
# 	return (float(N_sr*K_sr) / (mr*nr))

# def throughput(M,K,N,p,alpha,L,mr,nr):
# 	return float(M*N*K) / (num_cb_blks(M,K,N,p,alpha,L)*t_comp(p,alpha,L,mr,nr))

def cake_cpu_cache_sz(mc,kc,alpha,p):
	return (p*mc*kc*(1+alpha) + alpha*(p**2)*(mc**2))*4	

# note that mc will be fixed by L2 cache size...
# one must increase \alpha (N dim) or p to make use of more L3 cache
def num_cores(mc,L3,alpha):
	a = float(alpha*mc*mc)
	b = float(mc*mc*(1 + alpha))
	c = -float(L3/4.0)
	x = (-b + math.sqrt(b*b - 4*a*c)) / (2*a)
	return int(x)


def cb_shape_mc(mc,p,alpha):
	# mc = int(math.sqrt(float(L/4.0) / (2*(p + alpha*p) + alpha*p*p)))
	m = p*mc
	k = mc
	n = alpha*p*mc
	return (m, k, n)

def num_cb_blks(M,K,N,mc,p,alpha):
	M_sr, K_sr, N_sr = cb_shape_mc(mc,p,alpha)
	return(float(K)/K_sr * float(M)/M_sr * float(N)/N_sr)

def t_comp(mc,p,alpha,mr,nr):
	M_sr, K_sr, N_sr = cb_shape_mc(mc,p,alpha)
	return (float(N_sr*K_sr) / (mr*nr))

def throughput(M,K,N,mc,p,alpha,mr,nr):
	return float(M*N*K) / (num_cb_blks(M,K,N,mc,p,alpha)*t_comp(mc,p,alpha,mr,nr))


def plot_cake_goto_tput(alpha,M,N,K,mc,mr,nr,fname = 'cake_goto_tput'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['g','b','aqua','k','m','r']
	labels = ['CAKE - baseline DRAM bw', 'Goto - baseline DRAM bw',
	'CAKE - 1.5x DRAM', 'Goto - 1.5x DRAM']
	# local_mem = list(range(10,61,5))
	# local_mem = [10, 15, 20, 25, 30, 35, 40, 40, 45, 50, 55, 60]
	local_mem1 = [10, 15, 20, 25, 30, 35,40, 45, 50, 55, 60]
	l = list(range(1,41))
	p = [num_cores(mc,i*1e6,alpha) for i in l]
	cake_tput1 = [throughput(M,K,N,mc,i,alpha,mr,nr) for i in p]
	local_mem2 = [35, 40, 40, 45, 50, 55, 60]
	# cake_tput1 = [throughput(M,N,K,p,3,i*1e6,mr,nr) for i in local_mem1]
	# cake_tput2 = [throughput(M,N,K,p,1,i*1e6,mr,nr) for i in local_mem2]
	# goto_tput1 = [cake_tput[0]]*len(local_mem1)
	# goto_tput2 = [1.5*cake_tput[0]]*len(local_mem2)
	plt.figure(figsize = (6,4))
	plt.plot(l, cake_tput1, label = labels[0], color = '#FFC0CB') # need 2x the cache size to minimize evictions
	# plt.plot(local_mem1, goto_tput1, label = labels[1], color = 'dodgerblue') # need 2x the cache size to minimize evictions
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_goto_tput(1,10000,10000,10000,144,6,16)



def plot_power(ind = 'K', ntrials=10):
	if ind == 'M':
		xlab = "M (K=N=5000)"
	elif ind == 'K':
		xlab = "K (M=N=5000)"
	elif ind == 'N':
		xlab = "N (M=K=5000)"
	#
	labels = ['CAKE', 'MKL']
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	K = [10,50,100] + [i*500 for i in range(1,21)]
	# K = [i*50 for i in range(1,11)]
	cpu_power_cake_arr = []; dram_power_cake_arr = []; cpu_power_mkl_arr = []; dram_power_mkl_arr = []
	cpu_power_cake = 0; dram_power_cake = 0; cpu_power_mkl = 0; dram_power_mkl = 0
	gflops_mkl_arr=[];gflops_cake_arr=[];
	gflops_mkl = 0; gflops_cake = 0;
	dram_accs_cake_arr = [];  dram_accs_mkl_arr = []
	dram_accs_cake = 0; dram_accs_mkl = 0
	#
	for k in K:
		for j in range(1,ntrials+1):
			if ind == 'M':
				a = open("power_reports/cake_M%d-K5000-N5000-%d" % (k, j),'r').read().split('\n')
				b = open("power_reports/mkl_M%d-K5000-N5000-%d" % (k, j),'r').read().split('\n')
				df1 = pandas.read_csv('power_reports/cake_socw_M%d-K5000-N5000-%d.csv' % (k, j),skiprows=52,skipfooter=6)
				df2 = pandas.read_csv('power_reports/mkl_socw_M%d-K5000-N5000-%d.csv' % (k, j),skiprows=52,skipfooter=6)
				df1.columns = df1.columns.str.replace(' ', '')
				df2.columns = df2.columns.str.replace(' ', '')
				df3 = pandas.read_csv('results')
				lat_cake = df3[(df3['algo'] == 'cake') & (df3['M'] == k)]['time'].mean()  
				lat_mkl = df3[(df3['algo'] == 'mkl') & (df3['M'] == k)]['time'].mean()  
			elif ind == 'K':
				a = open("power_reports/cake_M5000-K%d-N5000-%d" % (k, j),'r').read().split('\n')
				b = open("power_reports/mkl_M5000-K%d-N5000-%d" % (k, j),'r').read().split('\n')
				df1 = pandas.read_csv('power_reports/cake_socw_M5000-K%d-N5000-%d.csv' % (k, j),skiprows=52,skipfooter=6)
				df2 = pandas.read_csv('power_reports/mkl_socw_M5000-K%d-N5000-%d.csv' % (k, j),skiprows=52,skipfooter=6)
				df1.columns = df1.columns.str.replace(' ', '')
				df2.columns = df2.columns.str.replace(' ', '')
				df3 = pandas.read_csv('results')
				lat_cake = df3[(df3['algo'] == 'cake') & (df3['K'] == k)]['time'].mean()  
				lat_mkl = df3[(df3['algo'] == 'mkl') & (df3['K'] == k)]['time'].mean()  
			elif ind == 'N':
				a = open("power_reports/cake_M5000-K5000-N%d-%d" % (k, j),'r').read().split('\n')
				b = open("power_reports/mkl_M5000-K5000-N%d-%d" % (k, j),'r').read().split('\n')
				df1 = pandas.read_csv('power_reports/cake_socw_M5000-K5000-N%d-%d.csv' % (k, j),skiprows=52,skipfooter=6)
				df2 = pandas.read_csv('power_reports/mkl_socw_M5000-K5000-N%d-%d.csv' % (k, j),skiprows=52,skipfooter=6)
				df1.columns = df1.columns.str.replace(' ', '')
				df2.columns = df2.columns.str.replace(' ', '')
				df3 = pandas.read_csv('results')
				lat_cake = df3[(df3['algo'] == 'cake') & (df3['N'] == k)]['time'].mean()  
				lat_mkl = df3[(df3['algo'] == 'mkl') & (df3['N'] == k)]['time'].mean()  
			#
			cpu_power_cake += float(a[5].split()[0])
			dram_power_cake += float(a[6].split()[0])
			gflops_cake += ((float(5000*5000*k) / float(a[8].split()[0]))  / (10**9))  
			dram_accs_cake += float(df1[df1['Device'] == 'Total ']['Total(bytes)']._values[0])
			# gflops_cake += ((float(500*500*k) / lat_cake)  / (10**9))  
			#
			cpu_power_mkl += float(b[5].split()[0])
			dram_power_mkl += float(b[6].split()[0])
			gflops_mkl += ((float(5000*5000*k) / float(b[8].split()[0]))  / (10**9))  
			dram_accs_mkl += float(df2[df2['Device'] == 'Total ']['Total(bytes)']._values[0])
			# gflops_mkl += ((float(500*500*k) / lat_mkl)  / (10**9))  
		#
		cpu_power_cake_arr.append(cpu_power_cake / ntrials)
		dram_power_cake_arr.append(dram_power_cake / ntrials)
		gflops_cake_arr.append(gflops_cake / ntrials)
		cpu_power_mkl_arr.append(cpu_power_mkl / ntrials)
		dram_power_mkl_arr.append(dram_power_mkl / ntrials)
		gflops_mkl_arr.append(gflops_mkl / ntrials)
		dram_accs_cake_arr.append(dram_accs_cake / ntrials)
		dram_accs_mkl_arr.append(dram_accs_mkl / ntrials)
		cpu_power_cake = 0; dram_power_cake = 0; cpu_power_mkl = 0; dram_power_mkl = 0
		gflops_mkl = 0; gflops_cake = 0;
		dram_accs_cake = 0; dram_accs_mkl = 0
	#
	#
	plt.figure(figsize = (6,4))
	plt.plot(K, cpu_power_cake_arr, label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(K, cpu_power_mkl_arr, label = labels[1], marker = markers[1], color = colors[1])
	plt.xticks(list(range(0,10001,2000)))
	# plt.yticks(list(range(0,1201,100)))
	plt.title('CPU Package Power Consumption')
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel("Energy (J)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("cpu_power_%s.pdf" % ind, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(K, dram_power_cake_arr, label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(K, dram_power_mkl_arr, label = labels[1], marker = markers[1], color = colors[1])
	plt.xticks(list(range(0,10001,2000)))
	# plt.yticks(list(range(0,41,5)))
	plt.title('DRAM Memory Power Consumption')
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel("Energy (J)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("dram_power_%s.pdf" % ind, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(K, [dram_power_cake_arr[i] + cpu_power_cake_arr[i] for i in range(len(K))], label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(K, [dram_power_mkl_arr[i] + cpu_power_mkl_arr[i] for i in range(len(K))], label = labels[1], marker = markers[1], color = colors[1])
	plt.xticks(list(range(0,10001,2000)))
	plt.title('CPU Package and DRAM Memory Power Consumption')
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel("Energy (J)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("total_power_%s.pdf" % ind, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(K, gflops_cake_arr, label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(K, gflops_mkl_arr, label = labels[1], marker = markers[1], color = colors[1])
	plt.xticks(list(range(0,10001,2000)))
	# plt.yticks(list(range(0,1201,100)))
	plt.title('Computation Throughput As %s Varies' % ind)
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "lower right", prop={'size': 12})
	plt.savefig("cpu_throughput_%s.pdf" % ind, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(K, dram_accs_cake_arr, label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(K, dram_accs_mkl_arr, label = labels[1], marker = markers[1], color = colors[1])
	plt.xticks(list(range(0,10001,2000)))
	plt.title('DRAM Accesses %s Varies' % ind)
	plt.xlabel(xlab, fontsize = 18)
	plt.ylabel("DRAM Accesses (bytes)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("dram_acc_%s.pdf" % ind, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_power(ind = 'M')
plot_power(ind = 'K')
plot_power(ind = 'N')
#





#------------ OLD PLOTS--------------#


# def plot_cake_vs_mkl_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_mkl'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal']
# 	NUM_CPUs = list(range(1,11))
# 	gflops_cpu=[];gflops_cake=[];dram_bw_cake=[];dram_bw_cpu=[];cake_mem_acc=[]
# 	#
# 	for i in range(len(NUM_CPUs)):
# 		df1 = pandas.read_csv('reports/report_mkl_%d.csv' % NUM_CPUs[i],skiprows=17,skipfooter=17)
# 		df2 = pandas.read_csv('reports/report_mkl_%d.csv' % NUM_CPUs[i],skipfooter=20)
# 		dram_bw_cpu.append(df1['Average']._values[0])
# 		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
# 		gflops_cpu.append((float(M*N*K) / cpu_time) / (10**9))
# 		#
# 		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % NUM_CPUs[i],skiprows=17,skipfooter=17)
# 		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % NUM_CPUs[i],skipfooter=20)
# 		dram_bw_cake.append(df1['Average']._values[0])
# 		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
# 		gflops_cake.append((float(M*N*K) / cpu_time) / (10**9))
# 		elapsed_time = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
# 		cake_mem_acc.append(cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,NUM_CPUs[i]) / elapsed_time)
# 	#
# 	# plt.subplot(1, 2, 1)
# 	plt.figure(figsize = (6, 4))
# 	plt.plot(NUM_CPUs, dram_bw_cpu, label = labels[1],  marker = markers[1], color = intel_color)
# 	plt.plot(NUM_CPUs, dram_bw_cake, label = labels[0],  marker = markers[0], color = colors[5])
# 	plt.plot(NUM_CPUs, cake_mem_acc, label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
# 	# plt.plot(list(NUM_CPUs), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
# 	#
# 	plt.title('(a) DRAM Bandwidth in CAKE vs MKL')
# 	plt.xlabel("Number of Cores", fontsize = 16)
# 	plt.xticks(NUM_CPUs)
# 	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 16)
# 	plt.legend(loc = "upper left", prop={'size': 11})
# 	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')
# 	#
# 	# plt.subplot(1, 2, 2)
# 	plt.figure(figsize = (6, 4))
# 	plt.plot(list(NUM_CPUs), list(gflops_cpu), label = labels[1],  marker = markers[3], color = intel_color)
# 	plt.plot(list(NUM_CPUs), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[5])
# 	#
# 	plt.title('(b) Computation Throughput in CAKE vs MKL')
# 	plt.xlabel("Number of Cores", fontsize = 16)
# 	plt.xticks(NUM_CPUs)
# 	plt.ylabel("Throughput (GFLOP/s)", fontsize = 16)
# 	plt.legend(loc = "upper left", prop={'size': 12})
# 	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')


# def plot_cake_vs_armpl_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_armpl'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['CAKE Observed', 'ARMPL Observed', 'CAKE Optimal']
# 	NUM_CPUs = [1,2,3,4]
# 	dram_bw_cpu=[];gflops_cpu=[];dram_bw_cake=[];gflops_cake=[];cake_mem_acc=[]
# 	#
# 	for i in range(len(NUM_CPUs)):
# 		a = open('reports_arm/report_arm_%d' % NUM_CPUs[i],'r').read().split('\n')
# 		cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
# 		dram_bw_cpu.append(((int(re.search(r'\d+', a[5]).group())*64.0) / cpu_time) / (10.0**9))
# 		gflops_cpu.append((float(M*N*K) / cpu_time) / (10**9))# / (float(NUM_CPUs[i]))
# 		#
# 		a = open('reports_arm/report_cake_%d' % NUM_CPUs[i],'r').read().split('\n')
# 		cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
# 		dram_bw_cake.append(((int(re.search(r'\d+', a[5]).group())*64.0) / cpu_time) / (10.0**9))
# 		gflops_cake.append((float(M*N*K) / cpu_time) / (10**9))# / (float(NUM_CPUs[i]))
# 		cake_mem_acc.append(cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,NUM_CPUs[i]) / cpu_time)
# 	#
# 	# plt.subplot(1, 2, 1)
# 	plt.figure(figsize = (6,4))
# 	plt.plot(list(NUM_CPUs), list(dram_bw_cpu), label = labels[1],  marker = markers[1], color = colors[4])
# 	plt.plot(list(NUM_CPUs), list(dram_bw_cake), label = labels[0],  marker = markers[0], color = colors[5])
# 	plt.plot(list(NUM_CPUs), list(cake_mem_acc), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
# 	#
# 	plt.title('(a) DRAM Bandwidth in CAKE vs ARMPL')
# 	plt.xlabel("Number of Cores", fontsize = 18)
# 	plt.xticks(NUM_CPUs)
# 	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
# 	plt.legend(loc = "center right", prop={'size': 10})
# 	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(list(NUM_CPUs), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[5])
# 	plt.plot(list(NUM_CPUs), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[4])
# 	#
# 	plt.ticklabel_format(useOffset=False, style='plain')
# 	plt.title('(b) Computation Throughput of CAKE vs ARMPL')
# 	plt.xlabel("Number of Cores", fontsize = 18)
# 	plt.xticks(NUM_CPUs)
# 	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
# 	plt.legend(loc = "upper left", prop={'size': 12})
# 	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
# 	# plt.suptitle('Performance of CAKE vs ARMPL', fontsize = 18)
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')


# def plot_cake_vs_amd_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_amd'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['CAKE Observed', 'OpenBlas Observed','CAKE Optimal']
# 	NUM_CPUs = list(range(1,17))
# 	dram_bw_cpu=[];gflops_cpu=[];dram_bw_cake=[];gflops_cake=[]
# 	#
# 	for i in range(len(NUM_CPUs)):
# 		a = open("reports/report_cake_amd_%d.csv" % NUM_CPUs[i],'r').read()
# 		data = [i for i in a.split('\n') if 'PID' in i][0].split(',')
# 		cpu_time = float(data[1]) / NUM_CPUs[i]
# 		dram_bw_cake.append(((float(data[2])*50000*64) / cpu_time) / (10**9))
# 		gflops_cake.append((float(M*N*K) / cpu_time) / (10**9))
# 		#
# 		a = open("reports/report_openblas_amd_%d.csv" % NUM_CPUs[i],'r').read()
# 		data = [i for i in a.split('\n') if 'PID' in i][0].split(',')
# 		cpu_time = float(data[1]) / NUM_CPUs[i]
# 		dram_bw_cpu.append(((float(data[2])*50000*64) / cpu_time) / (10**9))
# 		gflops_cpu.append((float(M*N*K) / cpu_time) / (10**9))
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(NUM_CPUs, dram_bw_cpu, label = labels[1],  marker = markers[1], color = colors[3])
# 	plt.plot(NUM_CPUs, dram_bw_cake, label = labels[0],  marker = markers[0], color = colors[5])
# 	#
# 	plt.title('(a) DRAM Bandwidth in CAKE vs OpenBlas')
# 	plt.xlabel("Number of Cores", fontsize = 18)
# 	plt.xticks(NUM_CPUs)
# 	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
# 	plt.legend(loc = "center right", prop={'size': 10})
# 	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(list(NUM_CPUs), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[5])
# 	plt.plot(list(NUM_CPUs), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[3])
# 	#
# 	plt.title('(b) Computation Throughput of CAKE vs OpenBlas')
# 	plt.xlabel("Number of Cores", fontsize = 18)
# 	plt.xticks(NUM_CPUs)
# 	plt.ylabel("Throughput (GFLOPS/sec)", fontsize = 18)
# 	plt.legend(loc = "lower right", prop={'size': 12})
# 	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
# 	# plt.show()
# 	# plt.clf()
# 	# plt.close('all')




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





# def plot_cake_vs_mkl_sparse(fname = 'cake_vs_mkl_sparse'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['CAKE Observed', 'MKL Observed','CAKE Optimal', 'MKL Optimal']
# 	density = [(1-i)*100 for i in [0.5,0.4,0.3,0.2,0.1,0.05,0.02,0.01,0.005]]
# 	N = 5000
# 	K = 2048
# 	M = 33708
# 	#
# 	df1 = pandas.read_csv('sparse.csv')
# 	cpu_times = df1['cpu_time']._values
# 	dram_bw_cpu = df1['bw']._values  # number of GB transferred b/w processors and DRAM
# 	gflops_cpu = [(float(M*N*K) / cpu_times[i]) / (10**9) for i in range(len(density))] # / (float(density[i]))
# 	#
# 	#
# 	dram_bw_cake = [7.284 for i in range(len(density))]
# 	gflops_cake = [(float(M*N*K) / 0.607994) / (10**9) for i in range(len(density))]
# 	dram_bw_cake_theo = [cake_cpu_DRAM_accesses(M,N,K,144,144,1,10) for i in density]
# 	#
# 	plt.plot(list(density), list(dram_bw_cpu), label = labels[1],  marker = markers[1], color = intel_color)
# 	# plt.plot(list(density), list(dram_bw_cake_theo), label = labels[2],  marker = markers[3], color = colors[5], linestyle = 'dashed')
# 	plt.plot(list(density), list(dram_bw_cake), label = labels[0],  marker = markers[0], color = colors[5])
# 	#
# 	plt.title('(a) DRAM Bandwidth for SparseMM in CAKE vs MKL')
# 	plt.xlabel("Percent Sparsity", fontsize = 18)
# 	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
# 	plt.legend(loc = "center left", prop={'size': 14})
# 	# plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')
# 	#
# 	plt.plot(list(density), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[5])
# 	plt.plot(list(density), list(gflops_cpu), label = labels[1],  marker = markers[3], color = intel_color)
# 	#
# 	plt.title('(b) Comp. Throughput of SparseMM in CAKE vs MKL')
# 	plt.xlabel("Percent Sparsity", fontsize = 18)
# 	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
# 	plt.legend(loc = "upper left", prop={'size': 14})
# 	# plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')
	



def plot_cap_flow():
	markers = ['o','v','s','d','^']
	colors = ['b','g','r','k','m']
	# csfont = {'fontname':'Times New Roman'}
	plt.rcParams["font.family"] = 'Arial'
	df1 = pandas.read_csv('cap_flow')
	years = df1['Year']._values
	plt.figure(figsize = (6,4))
	plt.plot(years, df1['Median Tax Revenue/GDP']._values, label = "Median Tax Revenue/GDP",  marker = markers[0], color = colors[0])
	plt.plot(years, df1['Median Net ODA/GDP']._values, label = "Median Net ODA/GDP",  marker = markers[1], color = colors[1])
	plt.plot(years, df1['Median Private Capital Flows/GDP']._values, label = "Median Private Capital\nInflows/GDP",  marker = markers[2], color = colors[2])
	plt.title('Median Private Capital Inflows versus\nMedian Tax Revenue and Median Net ODA',fontsize = 18)
	plt.xlabel("Year", fontsize = 14)
	plt.xticks(years)
	plt.xticks(rotation = 45)
	plt.ylabel("Median Flow/GDP %", fontsize = 14)
	# plt.legend(bbox_to_anchor=(2.35,.8), loc='center right', ncol=1, prop={'size': 10})
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.savefig('samplefigure', bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

plot_cap_flow()
