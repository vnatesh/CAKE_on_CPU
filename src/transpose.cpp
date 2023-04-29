#include "cake.h"



void transpose_mat(float* A_old, float* At, int K, int N, int Kt, int Nt) {
	
	int lda = N; 
	int ldb = K;
	int nt = (Nt/8)*8; int kt = (Kt/8)*8; int nr = 8; int kr = 8;
	nt = (nt == 0) ? 8 : nt;
	kt = (kt == 0) ? 8 : kt;

	for(int k = 0; k < K; k+=kt) { 

		#pragma omp parallel for
		for(int n = 0; n < N; n+=nt) { 
			for(int nn = n; nn < n + nt; nn+=nr) { 
				for(int kk = k; kk < k + kt; kk+=kr) { 
		
					if((nn + nr > N) || (kk + kr > K)) {
						transpose_mxn(&A_old[kk*N + nn], &At[nn*K + kk], lda, ldb, N - nn, K - kk);			
					} else {
						transpose_8x8(&A_old[kk*N + nn], &At[nn*K + kk], lda, ldb);
					}
				}
			}
		}
	}
}
