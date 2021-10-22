#include "cake.h"

int m_c;
int k_c;
int n_c;
int m_c1;
int k_c1;
int n_c1;
int m_c1_last_core;
int k_c1_last_core;
int mr_rem;
int nr_rem;
int k_rem;
int p_l;
int m_pad;
int k_pad;
int n_pad;
int Mb;
int Kb;
int Nb;
int M_padded;
int N_padded;

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
	ret->ncores = get_num_physical_cores();
	return ret;
}


int get_num_physical_cores() {

	FILE *fp;
	char ret1[16];
	char command1[128];

	sprintf(command1, "grep -c ^processor /proc/cpuinfo");
	fp = popen(command1, "r");

	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit(1);
	}

	if(fgets(ret1, sizeof(ret1), fp) == NULL) {
		printf("cpuinfo error\n");
	}
	

	char ret2[16];
	char command2[128];

	sprintf(command2, "lscpu | grep Thread | tr -dc '0-9'");
	fp = popen(command2, "r");

	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit(1);
	}

	if(fgets(ret2, sizeof(ret2), fp) == NULL) {
		printf("lscpu error\n");
	}
	

	pclose(fp);
	return atoi(ret1) / atoi(ret2);
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
			case 142:
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
	int max_threads = cake_cntx->ncores; // 2-way hyperthreaded
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




void init_block_dims(int M, int N, int K, int p, cake_cntx_t* cake_cntx, enum sched sch) {

	int m_r = cake_cntx->mr;
	int n_r = cake_cntx->nr;
	blk_dims_t* blk_dims = get_block_dims(cake_cntx, M, p, sch);
    m_c = blk_dims->m_c;
	k_c = blk_dims->k_c;
    n_c = blk_dims->n_c;

	switch(sch) {

		case KMN: {

			k_pad = (K % k_c) ? 1 : 0; 
			n_pad = (N % n_c) ? 1 : 0; 
			m_pad = (M % (p*m_c)) ? 1 : 0; 

			mr_rem = (int) ceil( ((double) (M % (p*m_c))) / m_r) ;
			int mr_per_core = (int) ceil( ((double) mr_rem) / p );
			
			if(mr_per_core) 
				p_l = (int) ceil( ((double) mr_rem) / mr_per_core);
			else
				p_l = 0;

			nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
			n_c1 = nr_rem * n_r;

			m_c1 = mr_per_core * m_r;
			m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;
			k_c1 = K % k_c;

			//number of CB blocks in the M, N, and K dims
			Mb = (M / (p*m_c)) + m_pad;
			Nb = (N / n_c) + n_pad;
			Kb = (K / k_c) + k_pad;

			M_padded = (m_r*mr_rem + (M /(p*m_c))*p*m_c);
			N_padded = (N - (N%n_c)) + n_c1;

			break;
		}


		case MKN: {

			k_pad = (K % (p*k_c)) ? 1 : 0; 
			m_pad = (M % m_c) ? 1 : 0; 
			n_pad = (N % n_c) ? 1 : 0;

			k_rem = K % (p*k_c);
			k_c1 = (int) ceil( ((double) k_rem) / p);

			if(k_c1) 
				p_l = (int) ceil( ((double) k_rem) / k_c1);
			else
				p_l = 0;

			nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
			n_c1 = nr_rem * n_r;

			k_c1_last_core = k_rem - k_c1*(p_l-1);
			mr_rem = (int) ceil( ((double) (M % m_c)) / m_r);
			m_c1 = mr_rem * m_r;

			// number of CB blocks in the M, N, and K dims
			Mb = (M / m_c) + m_pad;
			Kb = (K / (p*k_c)) + k_pad;
			Nb = (N / n_c) + n_pad;

			M_padded = (M / m_c)*m_c + m_c1;
			N_padded = (N - (N%n_c)) + n_c1;


			break;
		}

		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}

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


