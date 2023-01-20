#include "cake.h"



cake_cntx_t* cake_query_cntx_torch(int L2, int L3) {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha_n = 1.0;
    ret->alpha_n = alpha_n;
 
    // query block size for the microkernel
#ifdef USE_BLIS
    cntx_t* blis_cntx = bli_gks_query_cntx();
    ret->blis_cntx = (void*) blis_cntx;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);

#elif USE_CAKE_HASWELL
    ret->blis_cntx = NULL;
    ret->mr = 6;
    ret->nr = 16;

#elif USE_CAKE_ARMV8
    ret->blis_cntx = NULL;
    ret->mr = 8;
    ret->nr = 12;
#endif

	ret->L2 = L2;
	ret->L3 = L3;
	return ret;
}


cake_cntx_t* cake_query_cntx() {

    cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
    double alpha_n = 1.0;
    ret->alpha_n = alpha_n;
	ret->L1 = get_cache_size(1);
	ret->L2 = get_cache_size(2);
	ret->L3 = get_cache_size(3);
	ret->ncores = get_num_physical_cores();
	ret->peak_dram_bw = 32 * 1e9; // TODO : hardcoded bw and flops on i9 for now
	ret->peak_flops = 600 * 1e9;

    // query block size for the microkernel
#ifdef USE_BLIS
    cntx_t* blis_cntx = bli_gks_query_cntx();
    ret->blis_cntx = (void*) blis_cntx;
    ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
    ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);

#elif USE_CAKE_HASWELL
    ret->blis_cntx = NULL;
    ret->mr = 6;
    ret->nr = 16;
    ret->m_map = (ret->mr/2) - 1;
    ret->n_map = (ret->nr/16) - 1;

#elif USE_CAKE_ARMV8
    ret->blis_cntx = NULL;
    ret->mr = 8;
    ret->nr = 12;
    ret->m_map = (ret->mr/2) - 1;
    ret->n_map = (ret->nr/12) - 1;
#endif

	return ret;
}


void update_mr_nr(cake_cntx_t* cake_cntx, int m_r, int n_r) {
    cake_cntx->mr = m_r;
    cake_cntx->nr = n_r;
#ifdef USE_CAKE_HASWELL
    cake_cntx->m_map = (m_r/2) - 1;
    cake_cntx->n_map = (n_r/16) - 1;
#elif USE_CAKE_ARMV8
    cake_cntx->m_map = (m_r/2) - 1;
    cake_cntx->n_map = (n_r/12) - 1;
#endif
}


int get_num_physical_cores() {

	FILE *fp;
	char ret1[16];
	char command1[128];

	sprintf(command1, "grep -c ^processor /proc/cpuinfo");
	fp = popen(command1, "r");

	if (fp == NULL) {
		printf("Failed to run proc/cpuinfo command\n" );
		exit(1);
	}

	if(fgets(ret1, sizeof(ret1), fp) == NULL) {
		printf("cpuinfo error\n");
	}
	

	char ret2[16];
	char command2[128];

	sprintf(command2, "lscpu | grep Thread -m 1 | tr -dc '0-9'");
	fp = popen(command2, "r");

	if (fp == NULL) {
		printf("Failed to run lscpu1 command\n" );
		exit(1);
	}

	if(fgets(ret2, sizeof(ret2), fp) == NULL) {
		printf("lscpu error\n");
	}
	

	char ret3[16];
	char command3[128];

	sprintf(command3, "lscpu | grep Socket -m 1 | tr -dc '0-9'");
	fp = popen(command3, "r");

	if (fp == NULL) {
		printf("Failed to run lscpu1 command\n" );
		exit(1);
	}

	if(fgets(ret3, sizeof(ret3), fp) == NULL) {
		printf("lscpu error\n");
	}

	pclose(fp);
	return atoi(ret1) / (atoi(ret2)*atoi(ret3));
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
		printf("Failed to run lscpu2 command\n" );
		exit(1);
	}

	if(fgets(ret, sizeof(ret), fp) == NULL) {
		printf("lscpu error\n");
	}
	
	pclose(fp);
	model_id = atoi(ret);

	if(level == 1) {
		switch(model_id) {
			case 1:
				return (32 * (1 << 10));
			case 3:
				return (32 * (1 << 10));
			case 4:
				return (16 * (1 << 10));
			case 17:
				return (32 * (1 << 10));
			case 33:
				return (32 * (1 << 10));
			case 49:
				return (32 * (1 << 10));
			case 69:
				return (32 * (1 << 10));
			case 85:
				return (32 * (1 << 10));
			case 142:
				return (32 * (1 << 10));
			case 165:
				return (32 * (1 << 10));
			default:
				break;
		}
	}

	else if(level == 2) {
		switch(model_id) {
			case 1:
				return (512 * (1 << 10));
			case 3:
				return (32 * (1 << 10));
			case 4:
				return (16 * (1 << 10));
			case 17:
				return (512 * (1 << 10));
			case 33:
				return (512 * (1 << 10));
			case 49:
				return (512 * (1 << 10));
			case 69:
				return (256 * (1 << 10));
			case 85:
				return (1024 * (1 << 10));
			case 142:
				return (256 * (1 << 10));
			case 165:
				return (256 * (1 << 10));
			default:
				break;
		}
	}

	else if(level == 3) {
		switch(model_id) {
			case 1:
				return (64 * (1 << 20));
			case 3:
				return (1 * (1 << 20));
			case 4:
				return (512 * (1 << 10));
			case 17:
				return (4 * (1 << 20));
			case 33:
				return (64 * (1 << 20));
			case 49:
				return (128 * (1 << 20));
			case 69:
				return (4 * (1 << 20));
			case 85:
				return (36608 * (1 << 10));
			case 142:
				return (8 * (1 << 20));
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
		printf("Failed to run lscpu3 command\n" );
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


cache_dims_t* get_cache_dims(int M, int N, int K, int p, 
			cake_cntx_t* cake_cntx, enum sched sch, 
			char* argv[], float density, float type_size) {

	int mc, mc_ret, nc_ret, a, mc_L2 = 0, mc_L3 = 0;
	int max_threads = cake_cntx->ncores; // 2-way hyperthreaded
	int mn_lcm = lcm(cake_cntx->mr, cake_cntx->nr);
	// int mn_lcm = m_r;

	// solve for optimal mc,kc based on L2 size
	// L2_size >= 2*(mc*kc + kc*nr) + 2*(mc*nr)     (solve for x = m_c = k_c) 
	int b = 2*cake_cntx->nr;
	mc_L2 = (int)  ((-b + sqrt(b*b + 4*(((double) cake_cntx->L2) / (2*type_size)))) / 2.0) ;
	// mc_L2 -= (mc_L2 % mn_lcm);
	mc_L2 -= (mc_L2 % cake_cntx->mr);
	// printf("mc_L2 = %d\n", mc_L2);


	// solve for the optimal block size m_c and k_c based on the L3 size
	// L3_size >= 2*(p*mc*kc + alpha*p*mc*kc) + 2*(p*mc*alpha*p*mc)     (solve for x = m_c = k_c) 
	// We only use ~ half of the each cache to prevent our working blocks from being evicted
	// and to allow for double buffering of partial results in L3
	mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (2*type_size))  
			/ (max_threads * (1 + 1.0 + 1.0*max_threads)));
	mc_L3 -= (mc_L3 % cake_cntx->mr);
	// printf("mc_L3 = %d\n", mc_L3);

	// if mc_L3 is too small, reduce alpha. likewise if mc_L2 is too small, increase alpha
	// This will reduce/increase L3 tile size and utilize more/less DRAM bandwidth
	cake_cntx->alpha_n = ((double) mc_L3) / mc_L2;
	mc =  mc_L2;


	mc_ret = mc;
	if(M < p*cake_cntx->mr) {
		mc_ret = cake_cntx->mr;
	} else if(M < p*mc) {
		
		a = (M / p);
		if(a < cake_cntx->mr) {
			mc_ret = cake_cntx->mr;
		} else {
			a += (cake_cntx->mr - (a % cake_cntx->mr));
			mc_ret = a;
		}
	}

    cache_dims_t* blk_ret = (cache_dims_t*) malloc(sizeof(cache_dims_t));

	// set schedule to MEMA-derived optimal value or user-defined
	blk_ret->sch = (sch == NA ? 
					derive_schedule(M, N, K, p, mc_ret, cake_cntx) : 
					sch);


	// user-defined tile sizes
	int ss = 0;
	if(argv) {
		ss = atoi(argv[5]);
	}

	if(ss) {
		blk_ret->m_c = atoi(argv[6]);
		blk_ret->k_c = atoi(argv[7]);
		blk_ret->n_c = atoi(argv[8]);
	// sparsity-aware tiling when A matrix is sparse
	} else if(density > 0.0000001) {
		
		printf("sparsity-aware tiling\n");
		double a_coeff = (density/cake_cntx->mr) * ((int) ceil(density * cake_cntx->mr)) ;

		mc_L2 = (int)  ((-b + sqrt(b*b + 4*a_coeff*(((double) cake_cntx->L2) / (type_size)))) / (2.0*a_coeff)) ;
		mc_L2 -= (mc_L2 % cake_cntx->mr);

		mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (type_size))  
		/ (max_threads * (a_coeff + cake_cntx->alpha_n + cake_cntx->alpha_n*max_threads)));
		mc_L3 -= (mc_L3 % cake_cntx->mr);


		mc_ret = mc_L3;
		if(M < p*cake_cntx->mr) {
			mc_ret = cake_cntx->mr;
		} else if(M < p*mc) {
			
			a = (M / p);
			if(a < cake_cntx->mr) {
				mc_ret = cake_cntx->mr;
			} else {
				a += (cake_cntx->mr - (a % cake_cntx->mr));
				mc_ret = a;
			}
		}

		// spMM is always K-first so using nc_ret from KMN
		nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
		nc_ret -= (nc_ret % cake_cntx->nr);
		nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;

		blk_ret->m_c = mc_L3 < M ? mc_L3 : cake_cntx->mr;
		blk_ret->k_c = mc_L2 < K ? mc_L2 : K;
		blk_ret->n_c = nc_ret;

	// CAKE tiling for dense MM 
	} else {

		switch(blk_ret->sch) {

			case KMN: {
				nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
				nc_ret -= (nc_ret % cake_cntx->nr);
				nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;
				break;
			}

			case MKN: {
				nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
				nc_ret -= (nc_ret % cake_cntx->nr);
				nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;			
				break;
			}

			case NKM: {
				nc_ret = (int) mc_ret;
				nc_ret -= (nc_ret % cake_cntx->nr);
				nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;

				mc_ret = (int) (cake_cntx->alpha_n*mc_ret);
				mc_ret -= (mc_ret % cake_cntx->mr);
				mc_ret = mc_ret == 0 ? cake_cntx->mr : mc_ret;			
				break;
			}

			default: {
				printf("unknown schedule\n");
				exit(1);
			}
		}

		blk_ret->m_c = mc_ret;
		blk_ret->k_c = mc_ret;
		blk_ret->n_c = nc_ret;

		// blk_ret->m_c = mc_ret;
		// blk_ret->k_c = 200;
		// blk_ret->n_c = 1024;


		// if(cake_cntx->L3 >= 4*(M*K + K*N + M*N)) {

		// 	int div = M / p;
		// 	blk_ret->m_c = (div % cake_cntx->mr) ? (div + (cake_cntx->mr - (div % cake_cntx->mr))) : div;

		// 	int kc_max = ((32768 / type_size) - cake_cntx->mr*cake_cntx->nr) / (cake_cntx->mr + cake_cntx->nr);
		// 	blk_ret->k_c = K < kc_max ? K : kc_max;
		// 	blk_ret->n_c = (N % cake_cntx->nr) ? (N + (cake_cntx->nr - (N % cake_cntx->nr))) : N;
		// }
	}



	return blk_ret;
}



// derive and set schedule according to MEMA analysis
enum sched derive_schedule(int M, int N, int K, int p, 
					int mc_ret, cake_cntx_t* cake_cntx) {

	float m,k,n, K_cut_M, K_cut_N, N_cut_M, M_cut_N;

	m = (float) (p*mc_ret);
	k = (float) (p*mc_ret);
	n = (float) (p*mc_ret);

	K_cut_M = (2.0*M) / (1.0 + (M * ((2.0/k) - (1.0/m))));
	K_cut_N = (2.0*N) / (1.0 + (N * ((2.0/k) - (1.0/n))));

	// N/M dim cutoffs for M vs N choice
	m = (float) (cake_cntx->alpha_n*p*mc_ret);
	n = (float) (cake_cntx->alpha_n*p*mc_ret);
	N_cut_M = M / (1.0 + (M * ((1.0/n) - (1.0/m))));
	M_cut_N = N / (1.0 + (N * ((1.0/m) - (1.0/n))));

	// printf("K_cut_M %f K_cut_N %f N_cut_M %f M_cut_N %f\n",K_cut_M,
	//  K_cut_N,N_cut_M,M_cut_N);
	// IO optimal schedule based on input parameters M,K,N,m,k,n
	if((N <= N_cut_M) && (K <= K_cut_M)) {
		return MKN;
	} else if((M <= M_cut_N) && (K <= K_cut_N)) {
		return NKM;
	} else {
		return KMN;
	}
}


void init_block_dims(int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size) {

	int m_r = cake_cntx->mr;
	int n_r = cake_cntx->nr;
	cache_dims_t* cache_dims = get_cache_dims(M, N, K, p, 
									cake_cntx, sch, argv, density, type_size);
    x->m_c = cache_dims->m_c;
	x->k_c = cache_dims->k_c;
    x->n_c = cache_dims->n_c;
    x->sch = cache_dims->sch;
    free(cache_dims);
    
	switch(x->sch) {

		case KMN: {

			x->k_pad = (K % x->k_c) ? 1 : 0; 
			x->n_pad = (N % x->n_c) ? 1 : 0; 
			x->m_pad = (M % (p*x->m_c)) ? 1 : 0; 

			x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r) ;
			int mr_per_core = (int) ceil( ((double) x->mr_rem) / p );
			
			if(mr_per_core) 
				x->p_l = (int) ceil( ((double) x->mr_rem) / mr_per_core);
			else
				x->p_l = 0;

			x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
			x->n_c1 = x->nr_rem * n_r;

			x->m_c1 = mr_per_core * m_r;
			x->m_c1_last_core = (mr_per_core - (x->p_l*mr_per_core - x->mr_rem)) * m_r;
			x->k_c1 = K % x->k_c;

			//number of CB blocks in the M, N, and K dims
			x->Mb = (M / (p*x->m_c)) + x->m_pad;
			x->Nb = (N / x->n_c) + x->n_pad;
			x->Kb = (K / x->k_c) + x->k_pad;

			x->M_padded = (m_r*x->mr_rem + (M / (p*x->m_c))*p*x->m_c);
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;

			break;
		}


		case MKN: {

			x->k_pad = (K % (p*x->k_c)) ? 1 : 0; 
			x->m_pad = (M % x->m_c) ? 1 : 0; 
			x->n_pad = (N % x->n_c) ? 1 : 0;

			x->k_rem = K % (p*x->k_c);
			x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

			if(x->k_c1) 
				x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
			else
				x->p_l = 0;

			x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
			x->n_c1 = x->nr_rem * n_r;

			x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
			x->mr_rem = (int) ceil( ((double) (M % x->m_c)) / m_r);
			x->m_c1 = x->mr_rem * m_r;

			// number of CB blocks in the M, N, and K dims
			x->Mb = (M / x->m_c) + x->m_pad;
			x->Kb = (K / (p*x->k_c)) + x->k_pad;
			x->Nb = (N / x->n_c) + x->n_pad;

			x->M_padded = (M / x->m_c)*x->m_c + x->m_c1;
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;


			break;
		}


		case NKM: {

			x->k_pad = (K % (p*x->k_c)) ? 1 : 0; 
			x->m_pad = (M % (p*x->m_c)) ? 1 : 0; 
			x->n_pad = (N % x->n_c) ? 1 : 0;

			x->k_rem = K % (p*x->k_c);
			x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

			if(x->k_c1) 
				x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
			else
				x->p_l = 0;

			x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
			x->n_c1 = x->nr_rem * n_r;

			x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
			x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r);
			x->m_c1 = x->mr_rem * m_r;

			// number of CB blocks in the M, N, and K dims
			x->Mb = (M / (p*x->m_c)) + x->m_pad;
			x->Kb = (K / (p*x->k_c)) + x->k_pad;
			x->Nb = (N / x->n_c) + x->n_pad;

			x->M_padded = (M / (p*x->m_c))*(p*x->m_c) + x->m_c1;
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;

			break;
		}


		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}
}





int grid_dims_2d(int M, int N, int K, int p, int ncores) {

	int pn, low, high, pn_ret;
	
	// optimal pn
	pn = (int) roundf(sqrt(((double) p*N) / M));
	pn = (pn == 0) ? 1 : pn;
	pn = (pn > p) ? p : pn;


	// index into static table of factors to quickly find factor of p that is closest to optimal pn
	// table is created at library creation
	if(grid_dims[ncores - p][pn-1]) {
		pn_ret = pn;
		// pm = p / pn;
	} else {
		for(int i = pn; i < p; i++) {
			if(grid_dims[ncores - p][i]) {
				high = i;
				break;
			}
		}

		for(int i = (pn - 2); i > 0; i--) {
			if(grid_dims[ncores - p][i]) {
				low = i;
				break;
			}
		}

		pn_ret = ((high - (pn-1)) < ((pn-1) - low)) ? high + 1: low + 1;
		// pm = p / pn;
	}

	return (pn_ret < p ? pn_ret : p);
}




void init_block_dims_2d(int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size) {

	int m_r, n_r, ncores, pm, pn, low, high, 
		m_c, k_c, n_c, m_c1, k_c1, n_c1,
		mr_rem, nr_rem, m_c1_last_core, n_c1_last_core, pm_l, pn_l,
		kc_max, m_pad, k_pad, n_pad, M_padded, N_padded, 
		Mb, Kb, Nb, a, left_M, left_N, thresh_M, thresh_N;

	m_r = cake_cntx->mr;
	n_r = cake_cntx->nr;
	ncores = cake_cntx->ncores;

	// optimal pn
	pn = grid_dims_2d(M, N, K, p, ncores);
	pm = p / pn;

	M_padded = (M % m_r) ? (M + (m_r - (M % m_r))) : M;
	N_padded = (N % n_r) ? (N + (n_r - (N % n_r))) : N;

	// based on mr*kc + kc*nr + mr*nr <= L1 
	kc_max = ((cake_cntx->L1 / type_size) - m_r*n_r) / (m_r + n_r); 

	// based on mc*kc + nr*kc + mc*nc <= L2
	m_c = (int) ((-n_r + sqrt(1.0*n_r*n_r + 4.0*2*(((double) cake_cntx->L2) / 4.0))) / (2.0*2));
	m_c -= (m_c % m_r);

	// printf("Mb = %d, Nb = %d, k_c = %d, k_c1 = %d, pm = %d, pn = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
	// 	Mb, Nb, k_c, k_c1, pm, pn, m_c, n_c, m_c1, n_c1);


	if(M_padded == pm*cake_cntx->mr) {
		m_c = cake_cntx->mr;
		pm = 1;
	} else if(M_padded < pm*m_c) {
		
		a = M_padded / pm;
		if(a < cake_cntx->mr) {
			m_c = cake_cntx->mr;
			pm = M_padded / cake_cntx->mr;
		} else {
			a += ((a % cake_cntx->mr) ? (cake_cntx->mr - (a % cake_cntx->mr)) : 0);
			m_c = a;
		}
	}

	// printf("Mb = %d, Nb = %d, k_c = %d, k_c1 = %d, pm = %d, pn = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
	// 	Mb, Nb, k_c, k_c1, pm, pn, m_c, n_c, m_c1, n_c1);


	// we want mc = nc since square tile maximizes AI
	n_c = ((m_c % n_r) ? (m_c + (n_r - (m_c % n_r))) : m_c); 
	// k_c = (m_c > kc_max) ? kc_max : m_c;
	k_c = kc_max;
	k_c1 = K % k_c;


	if(N_padded == pn*cake_cntx->nr) {
		n_c = cake_cntx->nr;
		pn = 1;
	} else if(N_padded < pn*n_c) {
		
		a = N_padded / pn;
		if(a < cake_cntx->nr) {
			n_c = cake_cntx->nr;
			pn = N_padded / cake_cntx->nr;
		} else {
			a += ((a % cake_cntx->nr) ? (cake_cntx->nr - (a % cake_cntx->nr)) : 0);
			n_c = a;
		}
	}

	// printf("Mb = %d, Nb = %d, k_c = %d, k_c1 = %d, pm = %d, pn = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
	// 	Mb, Nb, k_c, k_c1, pm, pn, m_c, n_c, m_c1, n_c1);


	n_pad = (N % (pn*n_c)) ? 1 : 0; 
	m_pad = (M % (pm*m_c)) ? 1 : 0; 

	mr_rem = (int) ceil( ((double) (M % (pm*m_c))) / m_r) ;
	int mr_per_core = (int) ceil( ((double) mr_rem) / pm );
	
	if(mr_per_core) 
		pm_l = (int) ceil( ((double) mr_rem) / mr_per_core);
	else
		pm_l = 0;

	m_c1 = mr_per_core * m_r;
	m_c1_last_core = (mr_per_core - (pm_l*mr_per_core - mr_rem)) * m_r;


// printf("m_c1_last_core = %d mr_rem = %d, mr_per_core = %d\n", m_c1_last_core, mr_rem, mr_per_core);


	nr_rem = (int) ceil( ((double) (N % (pn*n_c))) / n_r) ;
	int nr_per_core = (int) ceil( ((double) nr_rem) / pn );
	
	if(nr_per_core) 
		pn_l = (int) ceil( ((double) nr_rem) / nr_per_core);
	else
		pn_l = 0;

	n_c1 = nr_per_core * n_r;
	n_c1_last_core = (nr_per_core - (pn_l*nr_per_core - nr_rem)) * n_r;

// printf("n_c1_last_core = %d nr_rem = %d, nr_per_core = %d\n", n_c1_last_core, nr_rem, nr_per_core);
	m_pad = (M % (pm*m_c)) ? 1 : 0; 
	k_pad = (K % k_c) ? 1 : 0;
	n_pad = (N % (pn*n_c)) ? 1 : 0; 


	//number of CB blocks in the M, N, and K dims
	Mb = (M / (pm*m_c)) + m_pad;
	Nb = (N / (pn*n_c)) + n_pad;
	Kb = (K / k_c) + k_pad;

	M_padded = (m_r*mr_rem + (M / (pm*m_c))*pm*m_c);
	N_padded = (n_r*nr_rem + (N / (pn*n_c))*pn*n_c);

    x->m_c = m_c;
	x->k_c = k_c;
    x->n_c = n_c;
    x->m_c1 = m_c1;
	x->k_c1 = k_c1;
    x->n_c1 = n_c1;
    x->m_c1_last_core = m_c1_last_core;
	x->n_c1_last_core = n_c1_last_core;
	x->mr_rem = mr_rem;
	x->nr_rem = nr_rem;
	x->pm_l = pm_l;
	x->pn_l = pn_l;
	x->pm = pm;
	x->pn = pn;
	x->m_pad = m_pad; 
	x->k_pad = k_pad; 
	x->n_pad = n_pad; 
	x->Mb = Mb;
	x->Kb = Kb;
	x->Nb = Nb;
	x->M_padded = M_padded;
	x->N_padded = N_padded;
	x->sch = sch;

	printf("Mb = %d, Nb = %d, k_c = %d, k_c1 = %d, pm = %d, pn = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
		Mb, Nb, k_c, k_c1, pm, pn, m_c, n_c, m_c1, n_c1);
	// exit(1);
}





void init_block_dims_2d_small(int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size) {

	int m_r, n_r, ncores, pm, pn, 
		m_c, k_c, n_c, m_c1, k_c1, n_c1,
		kc_max, m_pad, k_pad, n_pad, M_padded, N_padded, 
		Mb, Kb, Nb;

	m_r = cake_cntx->mr;
	n_r = cake_cntx->nr;
	ncores = cake_cntx->ncores;

	// optimal pn
	pn = grid_dims_2d(M, N, K, p, ncores);
	pm = p / pn;

	M_padded = (M % m_r) ? (M + (m_r - (M % m_r))) : M;
	N_padded = (N % n_r) ? (N + (n_r - (N % n_r))) : N;
	
	m_c = (M_padded / pm);
	m_c += (m_r - (m_c % m_r));

	n_c = (N_padded / pn);
	n_c += (n_r - (n_c % n_r));

	// based on mr*kc + kc*nr + mr*nr <= L1 
	kc_max = ((cake_cntx->L1 / type_size) - m_r*n_r) / (m_r + n_r); 
	k_c = K < kc_max ? K : kc_max;
	k_c = (m_c > kc_max) ? kc_max : m_c;

	m_c1 = M_padded - (pm - 1)*m_c;
	n_c1 = N_padded - (pn - 1)*n_c;
	k_c1 = K % k_c;

	// printf("Kb = %d, k_c = %d, k_c1 = %d, pm = %d, pn = %d, mc = %d, nc = %d, mc1 = %d, nc1 = %d\n", 
	// 	Mb, k_c, k_c1, pm, pn, m_c, n_c, m_c1, n_c1);

	x->M_padded = M_padded;
	x->N_padded = N_padded;
    x->m_c = m_c;
	x->k_c = k_c;
    x->n_c = n_c;
    x->m_c1 = m_c1;
	x->k_c1 = k_c1;
    x->n_c1 = n_c1;
	x->k_pad = (K % x->k_c) ? 1 : 0; 
	x->n_pad = (N % x->n_c) ? 1 : 0; 
	x->m_pad = (M % x->m_c) ? 1 : 0; 
	x->Kb = (K / x->k_c) + x->k_pad;
	x->pm = pm;
	x->pn = pn;
	x->sch = sch;
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


