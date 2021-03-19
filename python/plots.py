import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os



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
	b_s = 2
	#
	oi_list = [1/8., 1/4., 1/2., 1, 2, 3, 4, 6, 8, 16, 32, 64]
	p = [min(P_max, b_s*x) for x in oi_list]
	plt.plot(oi_list, p,'-b', label  = "roofline", linewidth=4, color = 'black')
	plt.scatter([s2],[0.893], color = 'r', s=40)
	plt.scatter([s4],[1.904], color = 'b', s=40)
	plt.scatter([s8],[3.529], color = 'g', s=40)
	#
	plt.title('Roofline Model for Various Pod Sizes')
	plt.xscale('log', basex=2)
	plt.yscale('log', basey=2)
	plt.xlabel('Operational Intensity (tile mults / tile)', fontsize = 20)
	plt.ylabel('Performance (tile mults / cycle)', fontsize = 20)
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



# number of dram accesses for sgemm in bytes
def cake_cpu_DRAM_accesses(m,n,k,mc,kc,alpha,p):
	return (((float(m*n*k)/(alpha*p*mc) + float(m*n*k)/(p*mc) + m*n)) / float(10**9))*4	

def cake_cpu_cache_sz(mc,kc,alpha,p):
	return (p*mc*kc*(1+alpha) + alpha*(p**2)*(mc**2))*4	

# Goto algorithm doesn't shape the MMM block to adjust 
# for different number of cores p. Instead it shapes block to maximize cache usage
# while leaving approx half of the cache space empty to allow for data reuse and prevent evictions
def mkl_cpu_DRAM_accesses(m,n,k,mc,kc,nc):
	return (((float(m*k*n)/(nc)) + (n*k) + (float(m*n*k)/kc)) / float(10**9))*4	

def mkl_cpu_DRAM_accesses_float(m,n,k,mc,kc,nc):
	return (((float(m*k*n)/(nc)) + (n*k) + (float(m*n*k)/kc)) / float(10**9))*4	

def plot_cake_vs_mkl_cpu_theoretical(M,N,K,mc,kc,nc,p):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m']
	labels = ['CAKE', 'Intel i9', "mkl:cake dram access"]
	NUM_CPUs = [1,2,3,4,5,6,7,8,9,10]
	cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,K,mc,kc,nc,i) for i in NUM_CPUs]
	mkl_mem_acc = [mkl_cpu_DRAM_accesses(M,N,K,mc,kc,nc*i) for i in NUM_CPUs]
	#
	plt.plot(list(NUM_CPUs), [mkl_mem_acc[i]/cake_mem_acc[i] for i in range(len(NUM_CPUs))], marker = markers[0], color = colors[0])
	plt.title('CAKE_dgemm vs MKL (Theoretical)')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("MKL:Cake Ratio of DRAM Accesses", fontsize = 12)
	plt.legend(loc = "middle right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



def plot_cake_vs_mkl_cpu(M,N,K,mc,kc,alpha):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Theoretical', 'MKL Theoretical']
	NUM_CPUs = [1,2,3,4,5,6,7,8,9,10]
	cpu_mem_acc = [0]*len(NUM_CPUs)
	gflops_cpu = [0]*len(NUM_CPUs)
	#
	#
	for i in range(len(NUM_CPUs)):
		df1 = pandas.read_csv('reports/report_mkl_%d.csv' % NUM_CPUs[i],skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_mkl_%d.csv' % NUM_CPUs[i],skipfooter=20)
		avg_dram_bw = df1['Average']._values[0]
		# cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
		cpu_time = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
		gflops_cpu[i] = (float(M*N*K) / cpu_time) / (10**9) # / (float(NUM_CPUs[i]))
		cpu_mem_acc[i] = float(avg_dram_bw * cpu_time)  # number of GB transferred b/w processors and DRAM
	#
	#
	NUM_CPUs_cake = [1,2,3,4,5,6,7,8,9,10]
	# NUM_CPUs_cake = [2,4,5,8,10]
	cake_mem_acc = [0]*len(NUM_CPUs_cake)
	gflops_cake = [0]*len(NUM_CPUs_cake)
	#
	#
	for i in range(len(NUM_CPUs_cake)):
		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % NUM_CPUs_cake[i],skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % NUM_CPUs_cake[i],skipfooter=20)
		avg_dram_bw = df1['Average']._values[0]
		# cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs_cake[i])
		cpu_time = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0] 
		gflops_cake[i] = (float(M*N*K) / cpu_time) / (10**9) # / (float(NUM_CPUs_cake[i]))
		cake_mem_acc[i] = float(avg_dram_bw * cpu_time) 
	#
	#
	plt.plot(NUM_CPUs_cake, cake_mem_acc, label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(NUM_CPUs, cpu_mem_acc, label = labels[1],  marker = markers[1], color = colors[1])
	#
	NUM_CPUs_t = [1,2,3,4,5,6,7,8,9,10]
	cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,i) for i in NUM_CPUs_t]
	# mkl_mem_acc = [mkl_cpu_DRAM_accesses(M,N,K,mc,kc,960) for i in NUM_CPUs_t]
	plt.plot(NUM_CPUs_t, cake_mem_acc, label = labels[2], color = colors[0], linewidth = 2)
	# plt.plot(list(NUM_CPUs_t), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
	#
	plt.title('DRAM Accesses in CAKE vs MKL')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Gigabytes Transferred", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 10})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.plot(list(NUM_CPUs_cake), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[2])
	plt.plot(list(NUM_CPUs), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('Performance of CAKE vs MKL')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Performance (GFLOPS/sec)", fontsize = 12)
	plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




def plot_cake_vs_mkl_sparse():
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Theoretical', 'MKL Theoretical']
	density = [0.5,0.4,0.3,0.2,0.1,0.05,0.02,0.01,0.005]
	N = 5000
	K = 33708
	M = 1024
	p = 20 # num threads used by sparse
	#
	df1 = pandas.read_csv('sparse.csv')
	avg_dram_bw = df1['bw']._values
	cpu_times = df1['cpu_time']._values / p
	gflops_cpu = [((M*N*K) / cpu_times[i]) / (10**9) for i in range(len(density))] # / (float(density[i]))
	cpu_mem_acc = avg_dram_bw * cpu_times  # number of GB transferred b/w processors and DRAM
	#
	#
	cake_cpu_times = [0]*len(density)
	cake_mem_acc = [0]*len(density)
	gflops_cake = [0]*len(density)
	#
	#
	for i in range(len(density)):
		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % N, skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % N, skipfooter=25)
		avg_dram_bw = df1['Average']._values[0]
		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / 10
		cake_cpu_times[i] = cpu_time
		gflops_cake[i] = (float(M*N*K) / cpu_time) # / (float(density[i]))
		cake_mem_acc[i] = float(avg_dram_bw * cpu_time) 
	#
	plt.plot(list(density), list(cake_mem_acc), label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(list(density), list(cpu_mem_acc), label = labels[1],  marker = markers[1], color = colors[1])
	#
	# cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,K,mc,kc,960,i) for i in density]
	# cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,i,mc,kc,alpha,10) for i in density]
	# mkl_mem_acc = [mkl_cpu_DRAM_accesses(M,i,K,mc,kc,960) for i in density]
	# plt.plot(list(density), list(cake_mem_acc), label = labels[2], color = colors[0], linewidth = 2)
	# plt.plot(list(density), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
	#
	plt.title('Effect of Sparsity on Transformer Training DRAM Access')
	plt.xlabel("Density", fontsize = 12)
	plt.ylabel("Gigabytes Transferred", fontsize = 12)
	plt.legend(loc = "upper left", prop={'size': 10})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	#
	plt.plot(list(density), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[2])
	plt.plot(list(density), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('Effect of Sparsity on Transformer Training Performance')
	plt.xlabel("Density", fontsize = 12)
	plt.ylabel("Performance (GFLOPS/sec)", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#









def plot_cake_vs_mkl_transformer(p,mc,kc,alpha):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE Observed', 'MKL Observed','CAKE Theoretical', 'MKL Theoretical']
	batch_size = list(range(1000,10001,1000))
	cpu_times = [0]*len(batch_size)
	cpu_mem_acc = [0]*len(batch_size)
	gflops_cpu = [0]*len(batch_size)
	K = 33708
	M = 1024
	#
	for i in range(len(batch_size)):
		df1 = pandas.read_csv('reports/report_mkl_%d.csv' % batch_size[i],skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_mkl_%d.csv' % batch_size[i],skipfooter=25)
		avg_dram_bw = df1['Average']._values[0]
		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / p
		cpu_times[i] = cpu_time
		gflops_cpu[i] = (float(M*K*batch_size[i]) / cpu_time) # / (float(batch_size[i]))
		cpu_mem_acc[i] = float(avg_dram_bw * cpu_time)  # number of GB transferred b/w processors and DRAM
	#
	#
	cake_cpu_times = [0]*len(batch_size)
	cake_mem_acc = [0]*len(batch_size)
	gflops_cake = [0]*len(batch_size)
	#
	#
	for i in range(len(batch_size)):
		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % batch_size[i],skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % batch_size[i],skipfooter=25)
		avg_dram_bw = df1['Average']._values[0]
		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / p
		cake_cpu_times[i] = cpu_time
		gflops_cake[i] = (float(M*K*batch_size[i]) / cpu_time) # / (float(batch_size[i]))
		cake_mem_acc[i] = float(avg_dram_bw * cpu_time) 
	#
	plt.plot(list(batch_size), list(cake_mem_acc), label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(list(batch_size), list(cpu_mem_acc), label = labels[1],  marker = markers[1], color = colors[1])
	#
	# cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,K,mc,kc,960,i) for i in batch_size]
	# cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,i,mc,kc,alpha,10) for i in batch_size]
	# mkl_mem_acc = [mkl_cpu_DRAM_accesses(M,i,K,mc,kc,960) for i in batch_size]
	# plt.plot(list(batch_size), list(cake_mem_acc), label = labels[2], color = colors[0], linewidth = 2)
	# plt.plot(list(batch_size), list(mkl_mem_acc), label = labels[3], color = colors[1], linewidth = 2)
	#
	plt.title('DRAM Accesses In Transformer Training')
	plt.xlabel("Batch Size", fontsize = 12)
	plt.ylabel("Gigabytes Transferred", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 10})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	#
	plt.plot(list(batch_size), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[2])
	plt.plot(list(batch_size), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('Performance of Transformer')
	plt.xlabel("Batch Size", fontsize = 12)
	plt.ylabel("Performance (GFLOPS/sec)", fontsize = 12)
	plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#






def plot_cake_vs_armpl_cpu(M,N,K,mc,kc,alpha):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE', 'ARMPL']
	NUM_CPUs = [1,2,3,4]
	cpu_mem_acc = [0]*len(NUM_CPUs)
	gflops_cpu = [0]*len(NUM_CPUs)
	cake_mem_acc = [0]*len(NUM_CPUs)
	gflops_cake = [0]*len(NUM_CPUs)
	#
	#
	for i in range(len(NUM_CPUs)):
		a = open('reports_arm/report_arm_%d' % NUM_CPUs[i],'r').read().split('\n')
		cpu_mem_acc[i] = int(re.search(r'\d+', a[5]).group())
		cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
		gflops_cpu[i] = (float(M*N*K) / cpu_time) # / (float(NUM_CPUs[i]))
		#
		a = open('reports_arm/report_cake_%d' % NUM_CPUs[i],'r').read().split('\n')
		cake_mem_acc[i] = int(re.search(r'\d+', a[5]).group())
		cpu_time = float(re.search(r'\d+\.\d+', a[7]).group())
		gflops_cake[i] = (float(M*N*K) / cpu_time) # / (float(NUM_CPUs[i]))
	#
	plt.plot(list(NUM_CPUs), list(cake_mem_acc), label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(list(NUM_CPUs), list(cpu_mem_acc), label = labels[1],  marker = markers[1], color = colors[1])
	#
	cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,K,mc,kc,alpha,i) for i in NUM_CPUs]
	plt.plot(list(NUM_CPUs), list(cake_mem_acc), label = labels[2], color = colors[0], linewidth = 2)
	#
	plt.title('DRAM Accesses in CAKE vs ARMPL')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Gigabytes Transferred", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 10})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.plot(list(NUM_CPUs), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[2])
	plt.plot(list(NUM_CPUs), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('Performance of CAKE vs ARMPL')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Performance (GFLOPS/sec)", fontsize = 12)
	plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



def plot_internal_bw_cpu(exp, arr_size, file_name, ncores):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# labels = ['']	
	a = open(file_name,'r').read().split('\n')
	a = [i for i in a if exp in i]
	a = [i.split('\t') for i in a]
	a = [i for i in a if i[6] == 'areasize=%d' % arr_size]
	NUM_CPUs = range(1,ncores+1)
	int_bw = []
	for i in range(len(NUM_CPUs)):
		int_bw.append(float(a[i][-2][10:]) / (10**9))
	#
	plt.plot(list(NUM_CPUs), int_bw,  marker = markers[2], color = colors[1])
	plt.title('L2 Cache Bandwidth on ARM CPU')
	plt.xlabel("Number of Cores", fontsize = 12)
	plt.ylabel("Bandwidth (GB/s)", fontsize = 12)
	plt.xticks(NUM_CPUs)
	# plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

# plot_internal_bw_cpu('ScanWrite64PtrUnrollLoop', 2**24, 'stats.txt', 10)
plot_internal_bw_cpu('ScanWrite64PtrUnrollLoop', 2**18, 'stats_arm.txt', 4)





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
	plot_cake_vs_mkl_cpu(23040,23040,23040)
	plot_cake_vs_mkl_cpu(23040,23040,23040,96,96,1):
