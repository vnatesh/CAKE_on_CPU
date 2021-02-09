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




 
def plot_mem_size_R(fname = 'mem_size_R', NUM_SA = 64):
	# 100 linearly spaced numbers
	R = np.linspace(1.01,2,100)
	SZ_sr = (NUM_SA*R) / (R-1)
	plt.figure(figsize=(4,3)) 
	plt.plot(R,SZ_sr, 'r')
	# plt.title("Local memory size as a function of R")
	plt.xlabel("R", fontsize = 20)
	plt.ylabel("memory size (tiles)", fontsize = 20)
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




def plot_DRAM_access(M,K,N,tile_sz, fname = 'dram_acc'):
	plt.rcParams.update({'font.size': 12})
	# NUM_SA = [4,8,16,32,64,128,256]
	NUM_SA = [1,2,4,8]
	markers = ['o','v']
	colors = ['g','b'] 	
	# C = [(2,2),(4,2),(4,4),(8,4),(8,8),(16,8),(16,16)]
	# C = [(4,4),(8,4),(8,8),(16,8)]
	# C = [(8,8,8,2),(16,8,8,2),(16,16,8,2),(32,16,8,2)]
	C = [(8,8,8,1.25),(16,8,8,1.25),(16,16,16,1.125),(32,16,16,1.125)]
	labels = ['CAKE DRAM','Intel i9 DRAM', 'CAKE SRAM', 'i9 SRAM']
	mem_acc = [ext_mem_accesses(M,K,N,c[0],c[1],c[2],c[3],tile_sz) / (10**9) for c in C]
	mem_sz = [local_mem_size(c[0],c[1],c[2],c[3],tile_sz) for c in C]
	#
	NUM_CPUs = [1,2,3,4,5,6,7,8,9,10]
	cpu_mem_acc = [0]*len(NUM_CPUs)
	for i in NUM_CPUs:
		df1 = pandas.read_csv('reports/report_%d.csv' % i,skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_%d.csv' % i,skipfooter=20)
		avg_dram_bw = df1['Average']._values[0]
		elapsed_time = df2[df2['Metric Name']=='Elapsed Time']['Metric Value']._values[0]
		cpu_mem_acc[i-1] = (avg_dram_bw * elapsed_time)/4 # divide by 4 bytes to get number of 32bit floats transferred
	# cpu_mem_acc = [1633248996, 1548046440, 1270838124, 2028060840, 1592447772, 1208436252, 1764052920, 1372841184]
	#
	plt.plot(list(NUM_SA), list(mem_acc), label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(list(NUM_CPUs), list(cpu_mem_acc), label = labels[1],  marker = markers[1], color = colors[1])
	# plt.plot(list(NUM_SA), list(mem_sz), label = labels[2], marker = markers[0], color = colors[0])
	# plt.plot(list(NUM_CPUs),[20*(10**6)] * len(NUM_CPUs), label = labels[3], marker = markers[1], color = colors[1])
	#
	plt.title('DRAM Accesses in CAKE vs. Intel i9 CPU')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Number of Values Transferred (10^9)", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 12})
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



def plot_cpu_speedup(fname = 'cpu_speedup'):
	plt.rcParams.update({'font.size': 12})
	NUM_SA = [1,2,4,8,16]
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m']
	labels = ['CAKE', 'Intel i9', 'Ideal']
	# mem_acc = [ext_mem_accesses(M,K,N,c[0],c[1],s,R,tile_sz) / (10**9) for c in C]
	# mem_sz = [local_mem_size(c[0],c[1],s,R,tile_sz) for c in C]
	#
	NUM_CPUs = [1,2,3,4,5,6,7,8,9,10]
	cpu_elapsed_time = [0]*len(NUM_CPUs)
	# CAKE_elapsed_time = [71305460/2,19123775,10546984,6311018]
	# CAKE_elapsed_time = [50135100,25104475,16777327,9743581,8388875,4395388]	
	# CAKE_elapsed_time = [50135100,25104475,12583052,9001763,5243533]	
	# CAKE_elapsed_time = [50135100,25104475,12583052,8695457,5243533]	
	# CAKE_elapsed_time = [50135100,25104475,12583052,8558623,4719757]	
	# CAKE_elapsed_time = [151003278,76596190,35684702,18788206]	
	# CAKE_elapsed_time = [582219,325740,164047,136049,82573]	
	CAKE_elapsed_time = [412221,280535,147790,136049,82573]	
	for i in NUM_CPUs:
		df2 = pandas.read_csv('reports/report_%d.csv' % i,skipfooter=20)
		cpu_elapsed_time[i-1] = df2[df2['Metric Name']=='Elapsed Time']['Metric Value']._values[0]
	# cpu_mem_acc = [1633248996, 1548046440, 1270838124, 2028060840, 1592447772, 1208436252, 1764052920, 1372841184]
	#
	speedup_cpu = [max(cpu_elapsed_time) / i for i in cpu_elapsed_time]
	speedup_CAKE = [max(CAKE_elapsed_time) / i for i in CAKE_elapsed_time]
	plt.plot(list(NUM_SA), list(speedup_CAKE), label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(list(NUM_CPUs), list(speedup_cpu), label = labels[1],  marker = markers[1], color = colors[1])
	plt.plot(list(NUM_CPUs), list(NUM_CPUs), label = labels[2],  linewidth=2, color = colors[2])	
	# plt.plot(list(NUM_SA), list(mem_sz), label = labels[2], marker = markers[0], color = colors[0])
	# plt.plot(list(NUM_CPUs),[20*(10**6)] * len(NUM_CPUs), label = labels[3], marker = markers[1], color = colors[1])
	#
	plt.title('Speedup for MMM on CAKE vs Intel CPU')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Speedup", fontsize = 12)
	plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



def plot_cpu_dram_bw(fname = 'cpu_dram_bw'):
	plt.rcParams.update({'font.size': 12})
	# NUM_SA = [4,8,16,32,64,128,256]
	markers = ['o','v']
	colors = ['g','b'] 	
	# C = [(2,2),(4,2),(4,4),(8,4),(8,8),(16,8),(16,16)]
	# C = [(4,4),(8,4),(8,8),(16,8)]
	labels = ['MM Avg. DRAm bw', 'Peak DRAM bw']
	#
	NUM_CPUs = [1,2,3,4,5,6,7,8,9,10]
	avg_dram_bw = [0]*len(NUM_CPUs)
	for i in NUM_CPUs:
		df1 = pandas.read_csv('reports/report_%d.csv' % i,skiprows=17,skipfooter=17)
		avg_dram_bw[i-1] = df1['Average']._values[0]
	# cpu_mem_acc = [1633248996, 1548046440, 1270838124, 2028060840, 1592447772, 1208436252, 1764052920, 1372841184]
	#
	plt.plot(list(NUM_CPUs), list(avg_dram_bw), label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(list(NUM_CPUs), [40]*len(NUM_CPUs), label = labels[1],  color = colors[1],linewidth=2)
	# plt.plot(list(NUM_SA), list(mem_sz), label = labels[2], marker = markers[0], color = colors[0])
	# plt.plot(list(NUM_CPUs),[20*(10**6)] * len(NUM_CPUs), label = labels[3], marker = markers[1], color = colors[1])
	#
	plt.title('DRAM bw Utilization vs Num CPU Cores')
	plt.xlabel("Number of Cores", fontsize = 12)
	plt.ylabel("Average DRAM bw (GB/s)", fontsize = 12)
	plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


def plot_local_mem_size(M,K,N,s,R,tile_sz):
	plt.rcParams.update({'font.size': 12})
	# NUM_SA = [4,8,16,32,64,128,256]
	NUM_SA = [1,2,4,8]
	markers = ['o','v']
	colors = ['g','b'] 	
	# C = [(2,2),(4,2),(4,4),(8,4),(8,8),(16,8),(16,16)]
	C = [(4,4),(8,4),(8,8),(16,8)]
	labels = ['CAKE SRAM', 'i9 SRAM']
	mem_sz = [local_mem_size(M,K,N,c[0],c[1],s,R,tile_sz) for c in C]
	NUM_CPUs = [1,2,3,4,5,6,7,8]
	cpu_mem_acc = [1633248996, 1548046440, 1270838124, 2028060840, 1592447772, 1208436252, 1764052920, 1372841184]
	#
	plt.plot(list(NUM_SA), list(mem_sz), label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(list(NUM_CPUs),[20*(10**6)] * len(NUM_CPUs), label = labels[1], marker = markers[1], color = colors[1])
	#
	plt.title('SRAM Size in CAKE vs. Intel i9 CPU')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Bytes", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 12})
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')






def cake_cpu_DRAM_accesses(m,n,k,mc,kc,nc,p):
	return (((float(m*n*k)/nc + float(m*n*k)/(p*mc) + m*n)) / float(10**9))*8	

def mkl_cpu_DRAM_accesses(m,n,k,mc,kc,nc):
	return (((float(m*k*n)/nc) + (n*k) + (float(m*n*k)/kc)) / float(10**9))*8	


def plot_cake_vs_mkl_cpu_theoretical(M,N,K,mc,kc,nc,p):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m']
	labels = ['CAKE', 'Intel i9', "mkl:cake dram access"]
	NUM_CPUs = [1,2,3,4,5,6,7,8,9,10]
	cake_mem_acc = [cake_cpu_DRAM_accesses(M,N,K,mc,kc,nc,i) for i in NUM_CPUs]
	mkl_mem_acc = [mkl_cpu_DRAM_accesses(M,N,K,mc,kc,nc*i) for i in NUM_CPUs]
	plt.plot(list(NUM_CPUs), list(cake_mem_acc), label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(list(NUM_CPUs), list(mkl_mem_acc), label = labels[1],  marker = markers[1], color = colors[1])
	#
	plt.title('DRAM Accesses in CAKE_dgemm vs MKL (Theoretical)')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Gigabytes Transferred", fontsize = 12)
	plt.legend(loc = "middle right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
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

def plot_cake_vs_mkl_cpu_actual(M,N,K):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m']
	labels = ['CAKE', 'Intel i9']
	NUM_CPUs = [2,3,4,5,6,7,8,9,10]
	cpu_mem_acc = [0]*len(NUM_CPUs)
	gflops_cpu = [0]*len(NUM_CPUs)
	#
	for i in range(len(NUM_CPUs)):
		df1 = pandas.read_csv('reports/report_mkl_%d.csv' % NUM_CPUs[i],skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_mkl_%d.csv' % NUM_CPUs[i],skipfooter=20)
		avg_dram_bw = df1['Average']._values[0]
		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs[i])
		gflops_cpu[i] = (float(M*N*K) / cpu_time) # / (float(NUM_CPUs[i])
		cpu_mem_acc[i] = float(avg_dram_bw * cpu_time)  # number of GB transferred b/w processors and DRAM
	#
	NUM_CPUs_cake = [2,3,4,5,6,8,10]
	cake_mem_acc = [0]*len(NUM_CPUs_cake)
	gflops_cake = [0]*len(NUM_CPUs_cake)
	#
	for i in range(len(NUM_CPUs_cake)):
		df1 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % NUM_CPUs_cake[i],skiprows=17,skipfooter=17)
		df2 = pandas.read_csv('reports/report_cake_dgemm_%d.csv' % NUM_CPUs_cake[i],skipfooter=20)
		avg_dram_bw = df1['Average']._values[0]
		cpu_time = df2[df2['Metric Name'] == 'CPU Time']['Metric Value']._values[0] / float(NUM_CPUs_cake[i])
		gflops_cake[i] = (float(M*N*K) / cpu_time) # / (float(NUM_CPUs_cake[i])
		cake_mem_acc[i] = float(avg_dram_bw * cpu_time) 
	#
	plt.plot(list(NUM_CPUs_cake), list(cake_mem_acc), label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(list(NUM_CPUs), list(cpu_mem_acc), label = labels[1],  marker = markers[1], color = colors[1])
	#
	plt.title('DRAM Accesses in CAKE_dgemm vs MKL (Actual)')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Gigabytes Transferred", fontsize = 12)
	plt.legend(loc = "upper right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.plot(list(NUM_CPUs_cake), list(gflops_cake), label = labels[0],  marker = markers[2], color = colors[2])
	plt.plot(list(NUM_CPUs), list(gflops_cpu), label = labels[1],  marker = markers[3], color = colors[3])
	#
	plt.title('Performance of CAKE_dgemm vs MKL')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("Performance (GFLOPS/sec)", fontsize = 12)
	plt.legend(loc = "lower right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.plot(list(NUM_CPUs_cake), [mkl_mem_acc[NUM_CPUs.index(i)]/cake_mem_acc[NUM_CPUs_cake.index(i)] for i in NUM_CPUs_cake], marker = markers[0], color = colors[0])
	plt.title('CAKE_dgemm vs MKL (Actual)')
	plt.xlabel("Number of PEs", fontsize = 12)
	plt.ylabel("MKL:Cake Ratio of DRAM Accesses", fontsize = 12)
	plt.legend(loc = "middle right", prop={'size': 12})
	# plt.savefig("./plots/%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')







if __name__ == '__main__':
	plot_cake_vs_mkl_cpu_actual(23040,23040,23040)




