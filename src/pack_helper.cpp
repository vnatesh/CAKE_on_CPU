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
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
			M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);
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
			M_padded = (cake_cntx->mr*mr_rem + (M / (p*x->m_c))*p*x->m_c);
			break;
		}

		case MKN: {
			mr_rem = (int) ceil( ((double) (M % x->m_c)) / cake_cntx->mr);
			M_padded = (M / x->m_c)*x->m_c + mr_rem*cake_cntx->mr;
			break;
		}

		case NKM: {			
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr);
			M_padded = (cake_cntx->mr*mr_rem + (M / (p*x->m_c))*p*x->m_c);
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




void pack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			pack_C_single_buf_k_first(C, C_p, M, N, p, x, cake_cntx);
			break;
		}
		case MKN: {
			pack_C_single_buf_m_first(C, C_p, M, N, p, x, cake_cntx);
			break;
		}
		case NKM: {
			pack_C_single_buf_n_first(C, C_p, M, N, p, x, cake_cntx);
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}

}


void pack_B(float* B, float* B_p, int K, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			pack_B_k_first(B, B_p, K, N, x, cake_cntx);
			break;
		}
		case MKN: {
			pack_B_m_first(B, B_p, K, N, p, x, cake_cntx);
			break;
		}
		case NKM: {
			pack_B_n_first(B, B_p, K, N, p, x, cake_cntx);
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}



double pack_A(float* A, float* A_p, int M, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			return pack_A_single_buf_k_first(A, A_p, M, K, p, x, cake_cntx);
		}
		case MKN: {
			return pack_A_single_buf_m_first(A, A_p, M, K, p, x, cake_cntx); 
		}
		case NKM: {
			return pack_A_single_buf_n_first(A, A_p, M, K, p, x, cake_cntx);
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}





void pack_A_sp(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			return pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
		}
		case MKN: {
			return pack_A_sp_m_first(A, A_p, M, K, p, sp_pack, x, cake_cntx); 
		}
		case NKM: {
			return pack_A_sp_n_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}



void unpack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			unpack_C_single_buf_k_first(C, C_p, M, N, p, x, cake_cntx); 
			break;
		}
		case MKN: {
			unpack_C_single_buf_m_first(C, C_p, M, N, p, x, cake_cntx); 
			break;
		}
		case NKM: {
			unpack_C_single_buf_n_first(C, C_p, M, N, p, x, cake_cntx); 
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}


