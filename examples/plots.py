import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys







def plot_test_online_pack(fname = 'results1cake'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# labels = ['CAKE', 'ONLINE', "ONLINE_BLIS", "single buf", "blis"]
	labels = ['CAKE', 'new']
	df1 = pandas.read_csv('results1cake')
	cake_time = df1[(df1.algo == 0)]['time'].values
	cake_onl = df1[(df1.algo == 1)]['time'].values
	dims = df1[(df1.algo == 0)]['M'].values
	cake_time = [dims[i]**3 / cake_time[i] / 1e9 for i in range(len(cake_time))]
	cake_onl = [dims[i]**3 / cake_onl[i] / 1e9 for i in range(len(cake_onl))]
	plt.plot(dims, cake_time, label = labels[0], color = colors[0])
	plt.plot(dims, cake_onl, label = labels[1], color = colors[1])
	#
	plt.title('Throughput For Different Packing')
	plt.xlabel("M=N=K", fontsize = 24)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 24)
	plt.legend(loc = "lower right", prop={'size': 16})
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')	
	speedup1 = [cake_onl[i] / cake_time[i] for i in range(len(cake_time))]	
	print("speedup over cake = %f" %  gmean(speedup1))
	print(stats.describe(speedup1))



plot_test_online_pack()


