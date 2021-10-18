#include "cake.h"


// launching CAKE sgemm on asynchronous thread

double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA, bool packedB, float alpha, float beta) {

	if(M > 2*K)  {
		return cake_sgemm_k_first(A, B, C, M, N, K, p, cake_cntx, packedA, packedB, alpha, beta);
	} else if(K >= M) {
		return cake_sgemm_k_first(A, B, C, M, N, K, p, cake_cntx, packedA, packedB, alpha, beta);
	} else {
		return cake_sgemm_k_first(A, B, C, M, N, K, p, cake_cntx, packedA, packedB, alpha, beta);
	}
}


void* cake_sgemm_launch(void* inputs) {
	
    struct gemm_input* inp = (struct gemm_input*) inputs;    
    double ans = cake_sgemm(inp->A, inp->B, inp->C, inp->M, inp->N, inp->K, inp->p, 
    	inp->cake_cntx, inp->packedA, inp->packedB, inp->alpha, inp->beta);
	//sleep(4);
	pthread_exit(NULL);
}

