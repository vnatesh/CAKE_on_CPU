#include "cake.h"



int cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {

	int mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
	int M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);

	return (M_padded * K) * sizeof(float);
}



int cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {
	
	int nr_rem = (int) ceil( ((double) (N % x->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N % x->n_c)) + n_c1;

	return (K * N_padded) * sizeof(float);
}



int cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {

	int mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
	int M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);

	int nr_rem = (int) ceil( ((double) (N % x->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N%x->n_c)) + n_c1;

	return (M_padded * N_padded) * sizeof(float);
}



