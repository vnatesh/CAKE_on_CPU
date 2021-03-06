#include "cake.h"


void cake_dgemm(float* A, float* B, float* C, int M, int N, int K, int p, cake_cntx_t* cake_cntx) {
	// contiguous row-storage (i.e. cs_c = 1) or contiguous column-storage (i.e. rs_c = 1). 
	// This preference comes from how the microkernel is most efficiently able to load/store 
	// elements of C11 from/to memory. Most microkernels use vector instructions to access 
	// contiguous columns (or column segments) of C11
	int m_c, k_c, n_c, m_r, n_r;	
	double alpha_n;
	float alpha, beta;
	struct timeval start, end;
	double diff_t;
	float** A_p; 
	float** C_p;
	float* B_p;
	inc_t rsc, csc;
	cntx_t* blis_cntx;
	// inc_t rsa, csa;
	// inc_t rsb, csb;

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx(M,N,K,p);
	}

	m_r = cake_cntx->mr;
	n_r = cake_cntx->nr;
	m_c = cake_cntx->mc;
	alpha_n = cake_cntx->alpha;
    blis_cntx = cake_cntx->blis_cntx;		

    if(DEBUG) printf("M = %d, N = %d, K = %d\n", M, N, K);
    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", m_r, n_r);

   //m_c = 48;
    k_c = m_c;
    // m_c = 12;
    // k_c = 6;
    n_c = (int) (alpha_n * p * m_c);
	omp_set_num_threads(p);

    if(DEBUG) printf("mc = %d, kc = %d, nc = %d\n", m_c, k_c, n_c);

	int k_pad = (K % k_c) ? 1 : 0; 
	int n_pad = (N % n_c) ? 1 : 0; 

	gettimeofday (&start, NULL);

	int mr_rem = (int) ceil( ((double) (M % (p*m_c))) / m_r) ;
	int mr_per_core = (int) ceil( ((double) mr_rem) / p );
	int p_l;
	if(mr_per_core) 
		p_l = (int) ceil( ((double) mr_rem) / mr_per_core);
	else
		p_l = 0;

	int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
	int n_c1 = nr_rem * n_r;
	int N_b = (N - (N%n_c)) + n_c1;


	// pack A
	gettimeofday (&start, NULL);

	A_p = (float**) malloc( (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l) * sizeof( float* ));
	pack_A(A, A_p, M, K, m_c, k_c, m_r, p);
	// exit(1);

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	if(DEBUG) printf("A pack time: %f \n", diff_t); 



	gettimeofday (&start, NULL);

	if(posix_memalign((void**) &B_p, 64, K * N_b * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}
	pack_B(B, B_p, K, N, k_c, n_c, n_r, alpha_n, m_c);

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	if(DEBUG) printf("B pack time: %f \n", diff_t); 



	gettimeofday (&start, NULL);

	C_p = (float**) malloc((((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad) * sizeof( float* ));
	pack_C(C, C_p, M, N, m_c, n_c, m_r, n_r, p, alpha_n);

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	if(DEBUG) printf("C pack time: %f \n", diff_t); 



	// #pragma omp parallel sections
	// {
	//     #pragma omp section
	//     {
	//     // pack A
	// 		A_p = (float**) malloc( (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l) * sizeof( float* ));
	// 		pack_A(A, A_p, M, K, m_c, k_c, m_r, p);
	// 		// exit(1);
	//     }
	//     #pragma omp section
	//     {
	// 		// pack B
	// 		if(posix_memalign((void**) &B_p, 64, K * N_b * sizeof(float))) {
	// 			printf("posix memalign error\n");
	// 			exit(1);
	// 		}
	// 		pack_B(B, B_p, K, N, k_c, n_c, n_r, alpha_n, m_c);

	//     }

	//     #pragma omp section
	//     {
	// 		C_p = (float**) malloc((((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad) * sizeof( float* ));
	// 		pack_C(C, C_p, M, N, m_c, n_c, m_r, n_r, p, alpha_n);
	// 		// if(beta_user > 0)  pack_C(C, C_p, M, N, m_c, n_c, m_r, n_r, p, alpha_n);
	//     }

	// }

	// gettimeofday (&end, NULL);
	// diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	// +end.tv_usec) - start.tv_usec) / (1000000.0);
	// if(DEBUG) printf("packing time: %f \n", diff_t); 

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;
    // rsc = 1; csc = m_r;
    rsc = n_r; csc = 1;
    // rsa = 1; csa = m_r;
    // rsb = n_r; csb = 1;

	// m = 6; n = 8; k = 8;
	//rsc = 1; csc = m;
	//rsa = 1; csa = m;
	//rsb = 1; csb = k;

	gettimeofday (&start, NULL);
	int n_reg, m_reg, m, k1, m1, n1;

	int m_c1 = mr_per_core * m_r;
	int m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;

	int k1_n = K / k_c;
	int k_c1 = K % k_c;


	// void (*blis_kernel)(dim_t, float*, float*, float*, 
	// 					float*, float*, inc_t, inc_t, 
	// 					auxinfo_t*, cntx_t*);
	// bli_sgemm_ukernel = bli_sgemm_haswell_asm_6x16;

	// compute largest portion of C that can be evenly partitioned into CBS blocks
	for(n1 = 0; n1 < (N / n_c); n1++) {
		for(m1 = 0; m1 < (M / (p*m_c)); m1++) {
			// pragma omp here (i_c loop)
			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p; m++) {
				for(k1 = 0; k1 < (K / k_c); k1++) {
					// pragma omp also here possible (j_r loop)
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {												
							bli_sgemm_ukernel(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}

				if(k_c1) {
					int k1_n = K / k_c;
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_sgemm_ukernel( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}
			}
		}

		// compute the final row of CBS result blocks each of shape p_l*m_c x n_c
		if((M % (p*m_c))) {

			m1 = (M / (p*m_c));

			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p_l; m++) {

				int m_cx = (m == (p_l - 1) ? m_c1_last_core : m_c1);

				for(k1 = 0; k1 < (K / k_c); k1++) {
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_cx / m_r); m_reg++) {		
							bli_sgemm_ukernel(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p_l + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}

				if(k_c1) {
				
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_cx / m_r); m_reg++) {
							bli_sgemm_ukernel( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p_l + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}
			}
		}
	}


	// compute last column of CBS result blocks of size m_c x n_c1
	n1 = (N / n_c);

	if(n_c1) {	
		for(m1 = 0; m1 < (M / (p*m_c)); m1++) {
			// pragma omp here (i_c loop)
			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p; m++) {
				for(k1 = 0; k1 < (K / k_c); k1++) {
					// pragma omp also here possible (j_r loop)
					// #pragma omp parallel num_threads(p)
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_sgemm_ukernel(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c1 + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}

				if(k_c1) {
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_sgemm_ukernel( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c1 + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}
			}
		}

		// compute the final CBS result block of shape p_l*m_c x n_c1
		if((M % (p*m_c))) {

			m1 = (M / (p*m_c));

			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p_l; m++) {

				int m_cx = (m == (p_l - 1) ? m_c1_last_core : m_c1);

				for(k1 = 0; k1 < (K / k_c); k1++) {
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_cx / m_r); m_reg++) {						
							bli_sgemm_ukernel(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p_l + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c1 + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}
				
				if(k_c1) {
				
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_cx / m_r); m_reg++) {
							bli_sgemm_ukernel( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p_l + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c1 + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, blis_cntx);
						}
					}
				}
			}
		}
	}
	

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 
	// exit(1);
	// print_packed_C(C_p, M, N, m_c, n_c);
	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p);

	gettimeofday (&start, NULL);

	unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 

	for(int i = 0; i < (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l); i++) {
		free(A_p[i]);
	}

	for(int i = 0; i < (((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad); i++) {
		free(C_p[i]);
	}

	free(A_p);
	free(B_p);
	free(C_p);
	free(cake_cntx);
}


