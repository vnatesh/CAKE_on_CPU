#include "cake.h"


cake_cntx_t* cake_query_cntx_torch(int L2, int L3) {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha = 1;

    // query block size for the microkernel
    cntx_t* blis_cntx = bli_gks_query_cntx();
    ret->blis_cntx = blis_cntx;
    ret->alpha = alpha;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);
	ret->L2 = L2;
	ret->L3 = L3;
	return ret;
}


cake_cntx_t* cake_query_cntx() {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha = 1;

    // query block size for the microkernel
    cntx_t* blis_cntx = bli_gks_query_cntx();
    ret->blis_cntx = blis_cntx;
    ret->alpha = alpha;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);
	ret->L2 = get_cache_size(2);
	ret->L3 = get_cache_size(3);
	return ret;
}


// find cache size at levels L1d,L1i,L2,and L3 using lscpu
int get_cache_size(int level) {

	int len, size = 0;
	FILE *fp;
	char ret[16];
	char command[128];

	sprintf(command, "lscpu --caches=NAME,ONE-SIZE \
					| grep L%d \
					| grep -Eo '[0-9]*M|[0-9]*K|0-9*G' \
					| tr -d '\n'", level);
	fp = popen(command, "r");

	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit(1);
	}

	if(fgets(ret, sizeof(ret), fp) == NULL) {
		printf("lscpu error\n");
		// quick hack for raspberry pi 3 cache sizes (32 KiB L1, 512 KiB L2 shared)
		if(level == 2) {
			return (32 * (1 << 10));
		} else if(level == 3) {
			return (512 * (1 << 10));
		}
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


int get_block_dim(cake_cntx_t* cake_cntx, int M, int p) {

	int mc_L2 = 0, mc_L3 = 0;
	int max_threads = omp_get_max_threads() / 2; // 2-way hyperthreaded
	int mn_lcm = lcm(cake_cntx->mr, cake_cntx->nr);
	// int mn_lcm = m_r;

	// solves for the optimal block size m_c and k_c based on the L3 size
	// L3_size >= p*mc*kc + 2*(kc*alpha*p*mc + p*mc*alpha*p*mc)     (solve for x = m_c = k_c) 
	// We only use ~ half of the each cache to prevent our working blocks from being evicted
	mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (sizeof(double)))  
			/ (max_threads * (1 + 2*cake_cntx->alpha + 2*cake_cntx->alpha*max_threads)));
	mc_L3 -= (mc_L3 % mn_lcm);

	// solves for optimal mc,kc based on L2 size
	// L2_size >= 2*(mc*kc + kc*nr) + mc*nr     (solve for x = m_c = k_c) 
	int b = 3*cake_cntx->nr;
	mc_L2 = (int)  (-b + sqrt(b*b + 4*2*(((double) cake_cntx->L2) / (sizeof(double))))) / (2*2)  ;
	mc_L2 -= (mc_L2 % mn_lcm);

	int mc = mc_L3 < mc_L2 ? mc_L3 : mc_L2;
	int a;

	if(M < p*cake_cntx->mr) {
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

