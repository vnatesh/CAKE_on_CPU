import subprocess
import sys
from functools import reduce


# get number of physical cores on system
def get_num_cores():
	ret1 = subprocess.check_output("grep -c ^processor /proc/cpuinfo", shell=True)
	ret2 = subprocess.check_output("lscpu | grep Thread -m 1 | tr -dc '0-9'", shell=True)
	ret3 = subprocess.check_output("lscpu | grep Socket -m 1 | tr -dc '0-9'", shell=True)
	return int(int(ret1) / (int(ret2)*int(ret3)))


def factors(n):    
    return sorted(set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))


def gen_factor_table(p, alpha):
	p_min = max(1,int(round(alpha*p)))
	p_max = p
	factor_list = [factors(i) for i in range(p_min, p_max+1)]
	ret = '''
#include "common.h"

static bool grid_dims[%d][%d] = \n{\n''' % (len(factor_list), p)
	table = [0]*len(factor_list)
	#
	for i in range(len(factor_list)):
		q = [0 for x in range(p)]
		for j in range(len(factor_list[i])):
			q[factor_list[i][j] - 1] = 1
		table[i] = q
	#
	table = list(reversed(table))
	for i in range(len(factor_list)):
		ret += '\t{' + ','.join(map(str,table[i])) + '},\n'
	f = open("include/tiling.h", 'r+')
	content = f.read()
	f.seek(0)
	f.write(ret[:-1] + '''
};\n\n''' + content)




if __name__ == '__main__':
	ncores = get_num_cores()
	print("Generating possible processing grids given %d cores" % ncores)
	gen_factor_table(ncores, 0.1)

