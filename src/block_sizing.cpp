#include "cake.h"


cake_cntx_t* cake_query_cntx_torch(int L2, int L3) {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha_n = 1.0;

    // query block size for the microkernel
    cntx_t* blis_cntx = bli_gks_query_cntx();
    ret->blis_cntx = blis_cntx;
    ret->alpha_n = alpha_n;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);
	ret->L2 = L2;
	ret->L3 = L3;
	return ret;
}


cake_cntx_t* cake_query_cntx() {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha_n = 1.0;

    // query block size for the microkernel
    cntx_t* blis_cntx = bli_gks_query_cntx();
    ret->blis_cntx = blis_cntx;
    ret->alpha_n = alpha_n;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);
	ret->L2 = get_cache_size(2);
	ret->L3 = get_cache_size(3);
	return ret;
}


// find cache size at levels L1d,L1i,L2,and L3 using lscpu
int get_cache_size(int level) {

	int model_id, len, size = 0;
	FILE *fp;
	char ret[16];
	char command[128];

	sprintf(command, "lscpu | grep Model \
					| head -1 \
					| tr -dc '0-9'");
	fp = popen(command, "r");

	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit(1);
	}

	if(fgets(ret, sizeof(ret), fp) == NULL) {
		printf("lscpu error\n");
	}
	
	pclose(fp);
	model_id = atoi(ret);

	if(level == 2) {
		switch(model_id) {
			case 3:
				return (32 * (1 << 10));
			case 69:
				return (256 * (1 << 10));
			case 165:
				return (256 * (1 << 10));
			default:
				break;
		}
	}

	if(level == 3) {
		switch(model_id) {
			case 3:
				return (1 * (1 << 20));
			case 69:
				return (4 * (1 << 20));
			case 165:
				return (20 * (1 << 20));
			default:
				break;
		}
	}


	if(level < 3) {
		sprintf(command, "lscpu --caches=NAME,ONE-SIZE \
						| grep L%d \
						| grep -Eo '[0-9]*M|[0-9]*K|0-9*G' \
						| tr -d '\n'", level);
		fp = popen(command, "r");
	} else {
		sprintf(command, "lscpu --caches=NAME,ALL-SIZE \
						| grep L%d \
						| grep -Eo '[0-9]*M|[0-9]*K|0-9*G' \
						| tr -d '\n'", level);
		fp = popen(command, "r");
	}

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


blk_dims_t* get_block_dims(cake_cntx_t* cake_cntx, int M, int p, enum sched sch) {

	int mc, mc_ret, nc_ret, a, mc_L2 = 0, mc_L3 = 0;
	int max_threads = omp_get_max_threads() / 2; // 2-way hyperthreaded
	int mn_lcm = lcm(cake_cntx->mr, cake_cntx->nr);
	// int mn_lcm = m_r;

	// solve for optimal mc,kc based on L2 size
	// L2_size >= 2*(mc*kc + kc*nr) + 2*(mc*nr)     (solve for x = m_c = k_c) 
	int b = 2*cake_cntx->nr;
	mc_L2 = (int)  (-b + sqrt(b*b + 4*(((double) cake_cntx->L2) / (2*sizeof(float))))) / 2 ;
	mc_L2 -= (mc_L2 % mn_lcm);

	// solve for the optimal block size m_c and k_c based on the L3 size
	// L3_size >= 2*(p*mc*kc + alpha*p*mc*kc) + 2*(p*mc*alpha*p*mc)     (solve for x = m_c = k_c) 
	// We only use ~ half of the each cache to prevent our working blocks from being evicted
	// and to allow for double buffering of partial results in L3
	mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (2*sizeof(float)))  
			/ (max_threads * (1 + cake_cntx->alpha_n + cake_cntx->alpha_n*max_threads)));
	mc_L3 -= (mc_L3 % mn_lcm);

	mc = mc_L3 < mc_L2 ? mc_L3 : mc_L2;

	mc_ret = mc;
	if(M < p*cake_cntx->mr) {
		mc_ret = mn_lcm;
	} else if(M < p*mc) {
		
		a = (M / p);
		if(a < mn_lcm) {
			mc_ret = mn_lcm;
		} else {
			a -= (a % mn_lcm);
			mc_ret = a;
		}
	}

    blk_dims_t* blk_ret = (blk_dims_t*) malloc(sizeof(blk_dims_t));

	switch(sch) {
		case KMN: {
			nc_ret = (int) cake_cntx->alpha_n*p*mc_ret;
			break;
		}
		case MKN: {
			nc_ret = (int) cake_cntx->alpha_n*p*mc_ret;
			break;
		}
		case NKM: {
			nc_ret = (int) cake_cntx->alpha_n*mc_ret;
			break;
		}
	}

	blk_ret->m_c = mc_ret;
	blk_ret->k_c = mc_ret;
	blk_ret->n_c = nc_ret;

	return blk_ret;
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


