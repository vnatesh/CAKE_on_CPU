#include "cake.h"
#include <immintrin.h>




double cake_sgemv(float* A, float* B, float* C, int M, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);

// default type size = 4 bytes for float
cache_dims_t* get_cache_dims_mvm(int M, int K, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float density = 0, float type_size = 4);

void init_block_dims_mvm(int M, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float density = 0, float type_size = 4);





bool cake_sgemv_checker(float* A, float* B, float* C, int M, int K) {

	float* C_check = (float*) calloc(M, sizeof( float ));

	#pragma omp parallel for
	for(int k = 0; k < K; k++) {
		for(int m = 0; m < M; m++) {
			C_check[m] += A[k*M + m] * B[k];
		}
		// printf("%f ", C_check[m*N + n]);
	}

	// #pragma omp parallel for
	// for(int m = 0; m < M; m++) {
	// 	for(int k = 0; k < K; k++) {
	// 		C_check[m] += A[m*K + k] * B[k];
	// 	}
	// 	// printf("%f ", C_check[m*N + n]);
	// }
	// printf("\n");

	// printf("\n\n\n\n\n");
	// exit(1);

    int CORRECT = 1;
    int cnt = 0;
    int ind1 = 0;
    float eps = 1e-3; // machine precision level

	for(int m = 0; m < M; m++) {
        // if(C_check[m1*N + n1] != C[ind1]) {
        if(fabs(C_check[ind1] - C[ind1]) > eps) {
            cnt++;
            CORRECT = 0;
        }

         // printf("%f\t%f\n", C_check[ind1], C[ind1]);
        ind1++; 
  	}
    

    //printf("\n\n");

	if(CORRECT) {
		printf("CORRECT!\n");
		return 0;
	} else {
		printf("WRONG!\n");
		printf("%d\n", cnt);
		return 1;
	}

	free(C_check);

	// for (int n1 = 0; n1 < N; n1++) {
 //      for (int m1 = 0; m1 < M; m1++) {
 //        printf("%f ", C_check[m1*N + n1]);
 //      }
 //    }
	// printf("\n\n\n\n");
}



// assumes A is col-major
void cake_sgemv_haswell_8x8_col(float* A, float* B, float* C, int m, int k, int M) {
			  
	__m256 a[8], b[8];
	__m256 c;
	
	c  = _mm256_loadu_ps(C);

	int rem = k % 8;
	k -= rem;

	// mvm unrolled 8 times
	for(int kk = 0; kk < k; kk += 8) { 

		a[0]  = _mm256_loadu_ps(A);
		a[1]  = _mm256_loadu_ps(A + M);
		a[2]  = _mm256_loadu_ps(A + 2*M);
		a[3]  = _mm256_loadu_ps(A + 3*M);
		a[4]  = _mm256_loadu_ps(A + 4*M);
		a[5]  = _mm256_loadu_ps(A + 5*M);
		a[6]  = _mm256_loadu_ps(A + 6*M);
		a[7]  = _mm256_loadu_ps(A + 7*M);

		b[0] = _mm256_broadcast_ss(B++);
		b[1] = _mm256_broadcast_ss(B++);
		b[2] = _mm256_broadcast_ss(B++);
		b[3] = _mm256_broadcast_ss(B++);
		b[4] = _mm256_broadcast_ss(B++);
		b[5] = _mm256_broadcast_ss(B++);
		b[6] = _mm256_broadcast_ss(B++);
		b[7] = _mm256_broadcast_ss(B++);
		
		c =  _mm256_fmadd_ps(a[0], b[0], c);
		c =  _mm256_fmadd_ps(a[1], b[1], c);
		c =  _mm256_fmadd_ps(a[2], b[2], c);
		c =  _mm256_fmadd_ps(a[3], b[3], c);
		c =  _mm256_fmadd_ps(a[4], b[4], c);
		c =  _mm256_fmadd_ps(a[5], b[5], c);
		c =  _mm256_fmadd_ps(a[6], b[6], c);
		c =  _mm256_fmadd_ps(a[7], b[7], c);

		A += 8*M;
	}
			
	for(int kk = 0; kk < rem; kk++) { 
		a[0]  = _mm256_loadu_ps(A);
		b[0] = _mm256_broadcast_ss(B++);
		c =  _mm256_fmadd_ps(a[0], b[0], c);
		A += M;	
	}
			
	_mm256_storeu_ps(C, c);
}



// assumes A is CAKE packed
void cake_sgemv_haswell_8x8(float* A, float* B, float* C, int m, int k) {
			  
	__m256 a[8], b[8];
	__m256 c;
	
	c  = _mm256_loadu_ps(C);

	int rem = k % 8;
	k -= rem;

	// mvm unrolled 8 times
	for(int kk = 0; kk < k; kk += 8) { 

		a[0]  = _mm256_loadu_ps(A);
		a[1]  = _mm256_loadu_ps(A + 8);
		a[2]  = _mm256_loadu_ps(A + 16);
		a[3]  = _mm256_loadu_ps(A + 24);
		a[4]  = _mm256_loadu_ps(A + 32);
		a[5]  = _mm256_loadu_ps(A + 40);
		a[6]  = _mm256_loadu_ps(A + 48);
		a[7]  = _mm256_loadu_ps(A + 56);

		b[0] = _mm256_broadcast_ss(B++);
		b[1] = _mm256_broadcast_ss(B++);
		b[2] = _mm256_broadcast_ss(B++);
		b[3] = _mm256_broadcast_ss(B++);
		b[4] = _mm256_broadcast_ss(B++);
		b[5] = _mm256_broadcast_ss(B++);
		b[6] = _mm256_broadcast_ss(B++);
		b[7] = _mm256_broadcast_ss(B++);
		
		c =  _mm256_fmadd_ps(a[0], b[0], c);
		c =  _mm256_fmadd_ps(a[1], b[1], c);
		c =  _mm256_fmadd_ps(a[2], b[2], c);
		c =  _mm256_fmadd_ps(a[3], b[3], c);
		c =  _mm256_fmadd_ps(a[4], b[4], c);
		c =  _mm256_fmadd_ps(a[5], b[5], c);
		c =  _mm256_fmadd_ps(a[6], b[6], c);
		c =  _mm256_fmadd_ps(a[7], b[7], c);

		A += 8*8;
	}
			
	for(int kk = 0; kk < rem; kk++) { 
		a[0]  = _mm256_loadu_ps(A);
		b[0] = _mm256_broadcast_ss(B++);
		c =  _mm256_fmadd_ps(a[0], b[0], c);
		A += 8;	
	}
			
	_mm256_storeu_ps(C, c);
}



typedef union {
    float vals[8];
    __m256 bits;
} m256;

// assumes A is row-major
void cake_sgemv_haswell_inner_1x64x1(float* A, float* B, float* C, int m, int k, int K) {
			  
	__m256 a[8], b[8];
	__m256 c;

    m256 tmp;
    m256 ans;

	c  = _mm256_loadu_ps(C);

	b[0] = _mm256_loadu_ps(B);
	b[1] = _mm256_loadu_ps(B + 8);
	b[2] = _mm256_loadu_ps(B + 16);
	b[3] = _mm256_loadu_ps(B + 24);
	b[4] = _mm256_loadu_ps(B + 32);
	b[5] = _mm256_loadu_ps(B + 40);
	b[6] = _mm256_loadu_ps(B + 48);
	b[7] = _mm256_loadu_ps(B + 56);

	// int rem = k % 8;
	// k -= rem;


	for(int mm = 0; mm < m; mm++) {

		for(int kk = 0; kk < k; kk += 64) { 

			a[0]  = _mm256_loadu_ps(A);
			a[1]  = _mm256_loadu_ps(A + 8);
			a[2]  = _mm256_loadu_ps(A + 16);
			a[3]  = _mm256_loadu_ps(A + 24);
			a[4]  = _mm256_loadu_ps(A + 32);
			a[5]  = _mm256_loadu_ps(A + 40);
			a[6]  = _mm256_loadu_ps(A + 48);
			a[7]  = _mm256_loadu_ps(A + 56);
			
			c =  _mm256_fmadd_ps(a[0], b[0], c);
			c =  _mm256_fmadd_ps(a[1], b[1], c);
			c =  _mm256_fmadd_ps(a[2], b[2], c);
			c =  _mm256_fmadd_ps(a[3], b[3], c);
			c =  _mm256_fmadd_ps(a[4], b[4], c);
			c =  _mm256_fmadd_ps(a[5], b[5], c);
			c =  _mm256_fmadd_ps(a[6], b[6], c);
			c =  _mm256_fmadd_ps(a[7], b[7], c);

			tmp.bits = c;
			ans.vals[mm] =  tmp.vals[0]+tmp.vals[1]+
							tmp.vals[2]+tmp.vals[3]+
							tmp.vals[4]+tmp.vals[5]+
							tmp.vals[6]+tmp.vals[7];
			A += 8*8;
		}

		A += (K-k);
	}

	// for(int kk = 0; kk < rem; kk++) { 
	// 	a[0]  = _mm256_loadu_ps(A);
	// 	b[0] = _mm256_broadcast_ss(B++);
	// 	c =  _mm256_fmadd_ps(a[0], b[0], c);
	// 	A += 8;	
	// }
			
	_mm256_storeu_ps(C, ans.bits);
}


cache_dims_t* get_cache_dims_mvm(int M, int K, int p, 
			cake_cntx_t* cake_cntx, enum sched sch, 
			char* argv[], float density, float type_size) {

	int mc, mc_ret, a, mc_L2 = 0, mc_L3 = 0;
	int max_threads = cake_cntx->ncores; // 2-way hyperthreaded

	// solve for optimal mc,kc based on L2 size
	// L2_size >= mc*kc + kc + mc     (solve for x = m_c = k_c) 
	int b = 2;
	mc_L2 = (int)  ((-b + sqrt(b*b + 4*(((double) cake_cntx->L2) / (type_size)))) / 2.0) ;
	// mc_L2 -= (mc_L2 % cake_cntx->mr);
	mc_L2 += (64 - (mc_L2 % 64));
	printf("mc_L2 = %d\n", mc_L2);


	// solve for the optimal block size m_c and k_c based on the L3 size
	// L3_size >= 2*(p*mc*kc + alpha*p*mc*kc) + 2*(p*mc*alpha*p*mc)     (solve for x = m_c = k_c) 
	// We only use ~ half of the each cache to prevent our working blocks from being evicted
	// and to allow for double buffering of partial results in L3
	mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (2*type_size))  
			/ (max_threads * (1 + 1.0 + 1.0*max_threads)));
	mc_L3 -= (mc_L3 % cake_cntx->mr);
	// printf("mc_L3 = %d\n", mc_L3);

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

	// user-defined tile sizes
	int ss = 0;
	if(argv) {
		ss = atoi(argv[5]);
	}

	if(ss) {
		printf("user-defined tile sizes\n");
		blk_ret->m_c = atoi(argv[6]);
		blk_ret->k_c = atoi(argv[7]);

	// CAKE tiling for dense MM 
	} else {
		blk_ret->m_c = mc_ret;
		blk_ret->k_c = mc_ret;

		// blk_ret->m_c = 64;
		// blk_ret->k_c = 64;

	}


	return blk_ret;
}



void init_block_dims_mvm(int M, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size) {

	int m_r = cake_cntx->mr;
	cache_dims_t* cache_dims = get_cache_dims_mvm(M, K, p, 
									cake_cntx, sch, argv, density, type_size);
    x->m_c = cache_dims->m_c;
	x->k_c = cache_dims->k_c;
    x->sch = cache_dims->sch;
    free(cache_dims);
    
	switch(x->sch) {

		case KMN: {

			x->k_pad = (K % x->k_c) ? 1 : 0; 
			x->m_pad = (M % (p*x->m_c)) ? 1 : 0; 

			x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r) ;
			int mr_per_core = (int) ceil( ((double) x->mr_rem) / p );
			
			if(mr_per_core) 
				x->p_l = (int) ceil( ((double) x->mr_rem) / mr_per_core);
			else
				x->p_l = 0;

			x->m_c1 = mr_per_core * m_r;
			x->m_c1_last_core = (mr_per_core - (x->p_l*mr_per_core - x->mr_rem)) * m_r;
			x->k_c1 = K % x->k_c;

			//number of blocks in the M and K dims
			x->Mb = (M / (p*x->m_c)) + x->m_pad;
			x->Kb = (K / x->k_c) + x->k_pad;

			x->M_padded = (m_r*x->mr_rem + (M / (p*x->m_c))*p*x->m_c);

			break;
		}


		case MKN: {

			x->k_pad = (K % (p*x->k_c)) ? 1 : 0; 
			x->m_pad = (M % x->m_c) ? 1 : 0; 

			x->k_rem = K % (p*x->k_c);
			x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

			if(x->k_c1) 
				x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
			else
				x->p_l = 0;


			x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
			x->mr_rem = (int) ceil( ((double) (M % x->m_c)) / m_r);
			x->m_c1 = x->mr_rem * m_r;

			// number of blocks in the M and K dims
			x->Mb = (M / x->m_c) + x->m_pad;
			x->Kb = (K / (p*x->k_c)) + x->k_pad;

			x->M_padded = (M / x->m_c)*x->m_c + x->m_c1;

			break;
		}


		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}
}





void schedule_mvm(float* A_p, float* B_p, float* C_p, int M, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr;
	int m_map = cake_cntx->m_map;

	int m_c = x->m_c, k_c = x->k_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
	int Mb = x->Mb, Kb = x->Kb;
	int M_padded = x->M_padded;

	int m, k, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m_cb, p_used, core;

	for(m = 0; m < Mb; m++) {

		if((m == Mb - 1) && m_pad) {
			p_used = p_l;
			m_cb = m_r*mr_rem ; //M % (p*m_c);
		} else {
			p_used = p;
			m_cb = p_used*m_c;
		}

		// pragma omp here (i_c loop)
		#pragma omp parallel for private(core,k)
		for(core = 0; core < p_used; core++) {

			// These vars must be private to thread, 
			// otherwise out of bounds memory access possible
			int m_c_t, m_c_x, k_c_t, m_reg;

			if((m == Mb - 1) && m_pad) {
				m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
				m_c_x = m_c1;
			} else {
				m_c_t = m_c;
				m_c_x = m_c; 
			}

			// pragma omp also here possible (j_r loop)
			for(k = 0; k < Kb; k++) {
				
				k_c_t = k_c; 
				if((k == Kb - 1) && k_pad) {
					k_c_t = k_c1;
				}

				int a_ind = m*p*m_c + k*k_c*M + core*m_c_x;
				int b_ind = k*k_c;
				int c_ind = m*p*m_c + core*m_c_x;

				for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

					cake_sgemv_haswell_8x8_col(&A_p[a_ind + m_reg*m_r], 
									&B_p[b_ind], 
									&C_p[c_ind + m_reg*m_r], 
									m_r, k_c_t, M);
				}
			}
		}
	}
}


// void schedule_mvm(float* A_p, float* B_p, float* C_p, int M, int K, int p, 
// 	cake_cntx_t* cake_cntx, blk_dims_t* x) {

// 	// copy over block dims to local vars to avoid readibility ussiues with x->
// 	int m_r = cake_cntx->mr;
// 	int m_map = cake_cntx->m_map;

// 	int m_c = x->m_c, k_c = x->k_c;
// 	int m_c1 = x->m_c1, k_c1 = x->k_c1;
// 	int m_c1_last_core = x->m_c1_last_core;
// 	int mr_rem = x->mr_rem;
// 	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
// 	int Mb = x->Mb, Kb = x->Kb;
// 	int M_padded = x->M_padded;

// 	int m, k, m_start, m_end, m_inc, k_start, k_end, k_inc;
// 	int m_cb, p_used, core;

// 	for(m = 0; m < Mb; m++) {

// 		if((m == Mb - 1) && m_pad) {
// 			p_used = p_l;
// 			m_cb = m_r*mr_rem ; //M % (p*m_c);
// 		} else {
// 			p_used = p;
// 			m_cb = p_used*m_c;
// 		}

// 		// pragma omp here (i_c loop)
// 		#pragma omp parallel for private(core,k)
// 		for(core = 0; core < p_used; core++) {

// 			// These vars must be private to thread, 
// 			// otherwise out of bounds memory access possible
// 			int m_c_t, m_c_x, k_c_t, m_reg;

// 			if((m == Mb - 1) && m_pad) {
// 				m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
// 				m_c_x = m_c1;
// 			} else {
// 				m_c_t = m_c;
// 				m_c_x = m_c; 
// 			}

// 			// pragma omp also here possible (j_r loop)
// 			for(k = 0; k < Kb; k++) {
				
// 				k_c_t = k_c; 
// 				if((k == Kb - 1) && k_pad) {
// 					k_c_t = k_c1;
// 				}

// 				int a_ind = m*p*m_c*K + k*k_c + core*m_c_x*K;
// 				int b_ind = k*k_c;
// 				int c_ind = m*p*m_c + core*m_c_x;

// 				for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

// 					cake_sgemv_haswell_inner_1x64x1(&A_p[a_ind + m_reg*m_r*K], 
// 									&B_p[b_ind], 
// 									&C_p[c_ind + m_reg*m_r], 
// 									m_r, k_c_t, K);
// 				}
// 			}
// 		}
// 	}
// }




// void schedule_mvm(float* A_p, float* B_p, float* C_p, int M, int K, int p, 
// 	cake_cntx_t* cake_cntx, blk_dims_t* x) {

// 	// copy over block dims to local vars to avoid readibility ussiues with x->
// 	int m_r = cake_cntx->mr;
// 	int m_map = cake_cntx->m_map;

// 	int m_c = x->m_c, k_c = x->k_c;
// 	int m_c1 = x->m_c1, k_c1 = x->k_c1;
// 	int m_c1_last_core = x->m_c1_last_core;
// 	int mr_rem = x->mr_rem;
// 	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
// 	int Mb = x->Mb, Kb = x->Kb;
// 	int M_padded = x->M_padded;

// 	int m, k, m_start, m_end, m_inc, k_start, k_end, k_inc;
// 	int m_cb, p_used, core;

// 	for(m = 0; m < Mb; m++) {

// 		if((m == Mb - 1) && m_pad) {
// 			p_used = p_l;
// 			m_cb = m_r*mr_rem ; //M % (p*m_c);
// 		} else {
// 			p_used = p;
// 			m_cb = p_used*m_c;
// 		}

// 		// pragma omp here (i_c loop)
// 		#pragma omp parallel for private(core,k)
// 		for(core = 0; core < p_used; core++) {

// 			// These vars must be private to thread, 
// 			// otherwise out of bounds memory access possible
// 			int m_c_t, m_c_x, k_c_t, m_reg;

// 			if((m == Mb - 1) && m_pad) {
// 				m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
// 				m_c_x = m_c1;
// 			} else {
// 				m_c_t = m_c;
// 				m_c_x = m_c; 
// 			}

// 			// pragma omp also here possible (j_r loop)
// 			for(k = 0; k < Kb; k++) {
				
// 				k_c_t = k_c; 
// 				if((k == Kb - 1) && k_pad) {
// 					k_c_t = k_c1;
// 				}

// 				int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
// 				int b_ind = k*k_c;
// 				int c_ind = m*p*m_c + core*m_c_x;

// 				for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

// 					cake_sgemv_haswell_8x8(&A_p[a_ind + m_reg*m_r*k_c_t], 
// 									&B_p[b_ind], 
// 									&C_p[c_ind + m_reg*m_r], 
// 									m_r, k_c_t);
// 				}
// 			}
// 		}
// 	}
// }




// // assumes CAKE packed A
// double cake_sgemv(float* A, float* B, float* C, int M, int K, int p, 
// 	cake_cntx_t* cake_cntx, char* argv[], bool packedA, float alpha, 
// 	float beta, enum sched sch) {


// 	if(cake_cntx == NULL) {
// 		cake_cntx = cake_query_cntx();
// 	}

// 	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
// 	omp_set_num_threads(p);

// 	int A_sz;	
// 	struct timespec start, end, start1, end1;
// 	long seconds, nanoseconds;
// 	double diff_t, times;
// 	float *A_p;


// 	clock_gettime(CLOCK_REALTIME, &start1);

// 	// sch = x->sch;
// 	sch = KMN;
// 	init_block_dims_mvm(M, K, p, x, cake_cntx, sch, argv, 0);

//     if(DEBUG) printf("m_r = %d\n", cake_cntx->mr);
//     if(DEBUG) printf("mc = %d, kc = %d, alpha_n = %f\n", x->m_c, x->k_c, cake_cntx->alpha_n);

// 	if(packedA) {
// 		A_p = A;
// 	} else {

// 		clock_gettime(CLOCK_REALTIME, &start);

// 		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);
// 		if(posix_memalign((void**) &A_p, 64, A_sz)) {
// 			printf("posix memalign error\n");
// 			exit(1);
// 		}

// 		pack_A(A, A_p, M, K, p, x, cake_cntx, sch);

// 		clock_gettime(CLOCK_REALTIME, &end);
// 		seconds = end.tv_sec - start.tv_sec;
// 		nanoseconds = end.tv_nsec - start.tv_nsec;
// 		diff_t = seconds + nanoseconds*1e-9;
// 		if(DEBUG) printf("A pack time: %f \n", diff_t ); 
// 	}

// 	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
// 	// otherwise just allocate an empty C_p buffer

// 	clock_gettime(CLOCK_REALTIME, &start);

// 	schedule_mvm(A_p, B, C, M, K, p, cake_cntx, x);

//     clock_gettime(CLOCK_REALTIME, &end);
//     seconds = end.tv_sec - start.tv_sec;
//     nanoseconds = end.tv_nsec - start.tv_nsec;
//     diff_t = seconds + nanoseconds*1e-9;
// 	if(DEBUG) printf("GEMV time: %f \n", diff_t); 	// exit(1);

// 	times = diff_t;


//     clock_gettime(CLOCK_REALTIME, &end1);
//     seconds = end1.tv_sec - start1.tv_sec;
//     nanoseconds = end1.tv_nsec - start1.tv_nsec;
//     diff_t = seconds + nanoseconds*1e-9;
// 	if(DEBUG) printf("full gemv time: %f \n", diff_t); 	// exit(1);

// 	if(!packedA) free(A_p);
// 	free(x);

// 	return times;
// }



// assumes A is unpacked
double cake_sgemv(float* A, float* B, float* C, int M, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, float alpha, 
	float beta, enum sched sch) {


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);

	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;


	clock_gettime(CLOCK_REALTIME, &start1);

	// sch = x->sch;
	sch = KMN;
	init_block_dims_mvm(M, K, p, x, cake_cntx, sch, argv, 0);

    if(DEBUG) printf("m_r = %d\n", cake_cntx->mr);
    if(DEBUG) printf("mc = %d, kc = %d, alpha_n = %f\n", x->m_c, x->k_c, cake_cntx->alpha_n);

	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
	// otherwise just allocate an empty C_p buffer

	clock_gettime(CLOCK_REALTIME, &start);

	schedule_mvm(A, B, C, M, K, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMV time: %f \n", diff_t); 	// exit(1);

	times = diff_t;


    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemv time: %f \n", diff_t); 	// exit(1);

	free(x);

	return times;
}



int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, p;
	struct timespec start, end;
	double diff_t;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	p = atoi(argv[3]);

	printf("M = %d, K = %d, cores = %d\n", M,K,p);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * sizeof( float ));
	float* C = (float*) calloc(M, sizeof( float ));

	// initialize A and B
    srand(time(NULL));

	rand_init(A, M, K);
	rand_init(B, K, 1);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	cake_cntx->mr = 8;

	clock_gettime(CLOCK_REALTIME, &start);

	cake_sgemv(A, B, C, M, K, p, cake_cntx);

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sgemv time: %f \n", diff_t); 


	cake_sgemv_checker(A, B, C, M, K);
	
	free(A);
	free(B);
	free(C);
	free(cake_cntx);
	
	return 0;
}



