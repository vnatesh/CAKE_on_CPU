import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys

intel_color = '#0071c5'



def plot_cake_vs_mkl_shape(fname = 'cake_vs_mkl_shape'):
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
		plt.plot(NUM_CPUs, speedup_mkl, label = "%d (mkl)" % j, color = colors[int(j/1000) - 1],linestyle='dashed',)
		plt.plot(NUM_CPUs, speedup_cake, label = "%d (cake)" % j, color = colors[int(j/1000) - 1])
	#
	plt.title('(a) Speedup For Square Matrices in CAKE vs MKL')
	plt.xlabel("Number of Cores (M=N=K)", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Speedup", fontsize = 18)
	plt.legend(title="M=N=K", loc = "upper left", prop={'size': 10})
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	# plt.show()
	plt.clf()
	plt.close('all')


if __name__ == '__main__':
	plot_cake_vs_mkl_shape()
