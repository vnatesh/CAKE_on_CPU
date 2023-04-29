#include "kernels.h"



void transpose_mxn(float* A, float* B, int lda, int ldb, int m, int n) {

	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			B[i*ldb + j] = A[j*lda + i];
		}
	}
}


void transpose_8x8(float* A, float* B, int lda, int ldb) {

	__m256 row0 = _mm256_loadu_ps((A + 0 * lda));
	__m256 row1 = _mm256_loadu_ps((A + 1 * lda));
	__m256 row2 = _mm256_loadu_ps((A + 2 * lda));
	__m256 row3 = _mm256_loadu_ps((A + 3 * lda));
	__m256 row4 = _mm256_loadu_ps((A + 4 * lda));
	__m256 row5 = _mm256_loadu_ps((A + 5 * lda));
	__m256 row6 = _mm256_loadu_ps((A + 6 * lda));
	__m256 row7 = _mm256_loadu_ps((A + 7 * lda));

	__m256 t0, t1, t2, t3, t4, t5, t6, t7;
	__m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;

	t0 = _mm256_unpacklo_ps(row0, row1);
	t2 = _mm256_unpacklo_ps(row2, row3);
	t4 = _mm256_unpacklo_ps(row4, row5);
	t6 = _mm256_unpacklo_ps(row6, row7);
	t1 = _mm256_unpackhi_ps(row0, row1);
	t3 = _mm256_unpackhi_ps(row2, row3);
	t5 = _mm256_unpackhi_ps(row4, row5);
	t7 = _mm256_unpackhi_ps(row6, row7);

	tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
	tt1 = _mm256_shuffle_ps(t0, t2, 0xee);
	tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
	tt3 = _mm256_shuffle_ps(t1, t3, 0xee);
	tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
	tt5 = _mm256_shuffle_ps(t4, t6, 0xee);
	tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
	tt7 = _mm256_shuffle_ps(t5, t7, 0xee);

	row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
	row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
	row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
	row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
	row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
	row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
	row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
	row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

	_mm256_storeu_ps((B + 0 * ldb), row0);
	_mm256_storeu_ps((B + 1 * ldb), row1);
	_mm256_storeu_ps((B + 2 * ldb), row2);
	_mm256_storeu_ps((B + 3 * ldb), row3);
	_mm256_storeu_ps((B + 4 * ldb), row4);
	_mm256_storeu_ps((B + 5 * ldb), row5);
	_mm256_storeu_ps((B + 6 * ldb), row6);
	_mm256_storeu_ps((B + 7 * ldb), row7);
}







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
