#include "cake.h"



double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA, bool packedB, float alpha, float beta) {

	if(M > 2*K)  {
	    if(DEBUG) printf("MKN Cake Schedule\n");
		return cake_sgemm_m_first(A, B, C, M, N, K, p, cake_cntx, packedA, packedB, alpha, beta);
	} else if(K >= M) {
	    if(DEBUG) printf("KMN Cake Schedule\n");
		return cake_sgemm_k_first(A, B, C, M, N, K, p, cake_cntx, packedA, packedB, alpha, beta);
	} else {
		return cake_sgemm_k_first(A, B, C, M, N, K, p, cake_cntx, packedA, packedB, alpha, beta);
	}
}


// launching CAKE sgemm on asynchronous thread

void* cake_sgemm_launch(void* inputs) {
	
    struct gemm_input* inp = (struct gemm_input*) inputs;    
    double ans = cake_sgemm(inp->A, inp->B, inp->C, inp->M, inp->N, inp->K, inp->p, 
    	inp->cake_cntx, inp->packedA, inp->packedB, inp->alpha, inp->beta);
	//sleep(4);
	pthread_exit(NULL);
}

