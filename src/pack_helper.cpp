#include "cake.h"



int cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	int mr_rem, M_padded;

	switch(sch) {

		case KMN: {
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
			M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);
			break;
		}

		case MKN: {
			mr_rem = (int) ceil( ((double) (M % x->m_c)) / cake_cntx->mr);
			M_padded = (M / x->m_c)*x->m_c + mr_rem*cake_cntx->mr;
			break;
		}

		case NKM: {
			int m_c_tmp = (int) (cake_cntx->alpha_n*p*x->m_c);
			mr_rem = (int) ceil( ((double) (M % m_c_tmp)) / cake_cntx->mr);
			M_padded = (M / m_c_tmp)*m_c_tmp + mr_rem*cake_cntx->mr;
			break;
		}

		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}

	return (M_padded * K) * sizeof(float);
}



int cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {
	
	int nr_rem = (int) ceil( ((double) (N % x->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N % x->n_c)) + n_c1;

	return (K * N_padded) * sizeof(float);
}



int cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	int mr_rem, M_padded;

	switch(sch) {

		case KMN: {
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
			M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);
			break;
		}

		case MKN: {
			mr_rem = (int) ceil( ((double) (M % x->m_c)) / cake_cntx->mr);
			M_padded = (M / x->m_c)*x->m_c + mr_rem*cake_cntx->mr;
			break;
		}

		case NKM: {
			int m_c_tmp = (int) (cake_cntx->alpha_n*p*x->m_c);
			mr_rem = (int) ceil( ((double) (M % m_c_tmp)) / cake_cntx->mr);
			M_padded = (M / m_c_tmp)*m_c_tmp + mr_rem*cake_cntx->mr;
			break;
		}

		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}


	int nr_rem = (int) ceil( ((double) (N % x->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N % x->n_c)) + n_c1;

	return (M_padded * N_padded) * sizeof(float);
}



