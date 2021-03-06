#include "cake.h"


cake_cntx_t* cake_query_cntx(int M, int N, int K, int p) {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha = 1;

    // query block size for the microkernel
    cntx_t* cntx = bli_gks_query_cntx();
    ret->blis_cntx = cntx;
    ret->alpha = alpha;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
	ret->mc = get_block_dim(ret->mr, ret->nr, alpha, M, p); 
	return ret;
}


// find cache size at levels L1d,L1i,L2,and L3 using lscpu
int get_cache_size(const char* level) {

	int len, size = 0;
	FILE *fp;
	char ret[16];
	char command[128];

	sprintf(command, "lscpu --caches=NAME,ONE-SIZE \
					| grep %s \
					| grep -Eo '[0-9]*M|[0-9]*K|0-9*G' \
					| tr -d '\n'", level);
	fp = popen(command, "r");

	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit(1);
	}

	if(fgets(ret, sizeof(ret), fp) == NULL) {
		printf("lscpu error\n");
	}

	pclose(fp);

	len = strlen(ret) - 1;

	// set cache size variables
	if(ret[len] == 'K') {
		ret[len] = '\0';
		size = atoi(ret) * (1 << 10);
	} else if(ret[len] == 'M') {
		ret[len] = '\0';
		size = atoi(ret) * (1 << 20);
	} else if(ret[len] == 'G') {
		ret[len] = '\0';
		size = atoi(ret) * (1 << 30);
	}

	return size;
}


int get_block_dim(int m_r, int n_r, double alpha_n, int M, int p) {

	int mc_L2 = 0, mc_L3 = 0;
	// find L3 and L2 cache sizes
	int max_threads = omp_get_max_threads() / 2; // 2-way hyperthreaded

	char const* l2 = "L2";
	char const* l3 = "L3";
	int L2_size = get_cache_size(l2);
	int L3_size = get_cache_size(l3);
	int mn_lcm = lcm(m_r, n_r);
	// int mn_lcm = m_r;

	// solves for the optimal block size m_c and k_c based on the L3 size
	// L3_size >= p*mc*kc + 2*(kc*alpha*p*mc + p*mc*alpha*p*mc)     (solve for x = m_c = k_c) 
	// We only use ~ half of the each cache to prevent our working blocks from being evicted
	mc_L3 = (int) sqrt((((double) L3_size) / (sizeof(double)))  
							/ (max_threads * (1 + 2*alpha_n + 2*alpha_n*max_threads)));
	mc_L3 -= (mc_L3 % mn_lcm);

	// solves for optimal mc,kc based on L2 size
	// L2_size >= 2*(mc*kc + kc*nr) + mc*nr     (solve for x = m_c = k_c) 
	int b = 3*n_r;
	mc_L2 = (int)  (-b + sqrt(b*b + 4*2*(((double) L2_size) / (sizeof(double))))) / (2*2)  ;
	mc_L2 -= (mc_L2 % mn_lcm);

	int mc = mc_L3 < mc_L2 ? mc_L3 : mc_L2;
	int a;

	if(M < p*m_r) {
		return mn_lcm;
	} else if(M < p*mc) {
		
		a = (M / p);
		if(a < mn_lcm) {
			return mn_lcm;
		}

		a -= (a % mn_lcm);
		return a;
	}

	// return min of possible L2 and L3 cache block sizes
	return mc;
}



// least common multiple
int lcm(int n1, int n2) {
	int max = (n1 > n2) ? n1 : n2;
	while (1) {
		if (max % n1 == 0 && max % n2 == 0) {
			break;
		}
		++max;
	}
	return max;
}

