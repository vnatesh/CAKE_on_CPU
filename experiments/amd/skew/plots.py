import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys




def plot_cake_vs_opt_mk(fname = 'cake_skew'):
  plt.rcParams.update({'font.size': 16})
  markers = ['o','v','s','d','^']
  colors = ['b','g','aqua','k','m','r']
  df1 = pandas.read_csv('result_skew')
  M = range(500,20001,500)
  for j in range(200,2501,500):
    K = j
    N = j
    tput_cake_m = [float(2*i*N*K / 1e9) / df1[(df1['algo'] == 'm-first') \
      & (df1['M'] == i) & (df1['K'] == K) & (df1['N'] == N)]['time'].mean() for i in M]
    tput_cake_k = [float(2*i*N*K / 1e9) / df1[(df1['algo'] == 'k-first') \
      & (df1['M'] == i) & (df1['K'] == K) & (df1['N'] == N)]['time'].mean() for i in M]
    tput_opt = [float(2*i*N*K / 1e9) / df1[(df1['algo'] == 'opt') \
      & (df1['M'] == i) & (df1['K'] == K) & (df1['N'] == N)]['time'].mean() for i in M]
    tput_blis = [float(2*i*N*K / 1e9) / df1[(df1['algo'] == 'blis') \
      & (df1['M'] == i) & (df1['K'] == K) & (df1['N'] == N)]['time'].mean() for i in M]
    fig = plt.figure(figsize = (6,4))
    plt.title('(a) FP32 Throughput With Large M', fontsize = 20)
    # plt.plot(M, tput_cake_m, 'b', label = 'm-first', color = colors[4])
    plt.plot(M, tput_cake_k, 'b', label = 'k-first', color = colors[0])
    plt.plot(M, tput_opt, 'b', label = 'opt', color = colors[1])
    plt.plot(M, tput_blis, 'b', label = 'blis', color = colors[3])
    plt.legend( prop={'size': 16})
    plt.xlabel('M (K = %d, N = %d)' % (K,N), fontsize = 20)
    plt.ylabel('Throughput (GFLOPs/sec)', fontsize = 20)
    # plt.xticks(range(0,111,20),fontsize = 14)
    # plt.yticks(range(0,1001,200),fontsize = 20)
    plt.ylim(ymin = 0,ymax = 1600)
    # plt.ticklabel_format(axis="y", style="sci")
    plt.savefig("%s_%d.pdf" % (fname,K), bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close('all')
  

plot_cake_vs_opt_mk()


