#include <stdio.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include "blis.h"
 
void pack_B(double* B, double* B_p, int K, int N, int k_c, int n_c, int n_r, int alpha_n, int m_c);
void pack_A(double* A, double** A_p, int M, int K, int m_c, int k_c, int m_r, int p);
void pack_C(double** C_p, int M, int N, int m_c, int n_c, int p, int alpha_n);
void unpack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);
void rand_init(double* mat, int r, int c);
void cake_dgemm_checker(double* A, double* B, double* C, int N, int M, int K);
void cake_dgemm(double* A, double* B, double* C, int M, int N, int K, int p);

int get_block_dim(int m_r, int n_r, double alpha_n);
int get_cache_size(char* level);
int lcm(int n1, int n2);

void print_packed_A(double** A_p, int M, int K, int m_c, int k_c, int p);
void print_packed_C(double** C_p, int M, int N, int m_c, int n_c);

void unpack_C_rsc(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);


int main( int argc, char** argv ) {

    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

	int M, K, N, p;

	// M = 1111;
	// K = 1111;
	// N = 2880;
	M = 2777;
	K = 2777;
	N = 2777;


	// M = 96;
	// K = 14583;
	// N = 96;
	// m_c = 96;
	// k_c = 96;

	// M = 23040;
	// K = 23040;
	// N = 23040;

	// M = 960;
	// K = 960;
	// N = 960;

    p = atoi(argv[1]);
	printf("M = %d, K = %d, N = %d\n", M,K,N);
	double* A = (double*) malloc(M * K * sizeof( double ));
	double* B = (double*) malloc(K * N * sizeof( double ));
	double* C = (double*) calloc(M * N , sizeof( double ));

	// initialize A and B
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);

	cake_dgemm(A, B, C, M, N, K, p);
	// bli_dprintm( "C: ", M, N, C, N, 1, "%4.4f", "" );
	cake_dgemm_checker(A, B, C, N, M, K);

	free(A);
	free(B);
	free(C);

	return 0;
}



void cake_dgemm(double* A, double* B, double* C, int M, int N, int K, int p) {
	// contiguous row-storage (i.e. cs_c = 1) or contiguous column-storage (i.e. rs_c = 1). 
	// This preference comes from how the microkernel is most efficiently able to load/store 
	// elements of C11 from/to memory. Most microkernels use vector instructions to access 
	// contiguous columns (or column segments) of C11
	int m_c, k_c, n_c, m_r, n_r;	
	double alpha, beta, alpha_n;
	struct timeval start, end;
	double diff_t;
	inc_t rsa, csa;
	inc_t rsb, csb;
	inc_t rsc, csc;

    // query block size for the microkernel
    cntx_t* cntx = bli_gks_query_cntx();
    m_r = (int) bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    n_r = (int) bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    printf("m_r = %d, n_r = %d\n\n", m_r, n_r);

    alpha_n = 1;
    m_c = get_block_dim(m_r, n_r, alpha_n);
    k_c = m_c;

    // m_c = 24;
    // k_c = 6;
    n_c = (int) (alpha_n * p * m_c);
	omp_set_num_threads(p);

    printf("mc = %d, kc = %d, nc = %d\n",m_c,k_c,n_c );

	int k_pad = (K % k_c) ? 1 : 0; 
	int m_pad = (M % m_c) ? 1 : 0; 
	int n_pad = (N % n_c) ? 1 : 0; 

    // pack A
	// double** A_p = (double**) malloc(p * ((K+k_pad)/k_c) * ((M+m_pad)/(p*m_c)) * sizeof( double* ));
	double** A_p = (double**) malloc( (K/k_c + k_pad)  * (M/m_c + m_pad) * sizeof( double* ));

	pack_A(A, A_p, M, K, m_c, k_c, m_r, p);
	// print_packed_A(A_p, M, K, m_c, k_c, p);

	// exit(1);

	// pack B
	int x = (int) (alpha_n * m_c);
	int q = (N/x)*x + m_c; 
	printf("x = %d\n", x );
	printf("q = %d \n", q);

	// exit(1);
	double* B_p;
	int ret;
	ret = posix_memalign((void**) &B_p, 64, K * q * sizeof(double));
	if(ret) {
		printf("posix memalign error\n");
		exit(1);
	}
	pack_B(B, B_p, K, N, k_c, n_c, n_r, alpha_n, m_c);


	// pack C
	double** C_p = (double**) malloc((M/m_c + m_pad) * (N/n_c + n_pad) * sizeof( double* ));
	pack_C(C_p, M, N, m_c, n_c, p, alpha_n);

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;
    // rsc = 1; csc = m_r;
    rsc = n_r; csc = 1;
    rsa = 1; csa = m_r;
    rsb = n_r; csb = 1;

	// m = 6; n = 8; k = 8;
	//rsc = 1; csc = m;
	//rsa = 1; csa = m;
	//rsb = 1; csb = k;

	gettimeofday (&start, NULL);
	int n_reg, m_reg, m, k1, m1, p_l, n1, n_c1, p_ln;


	// compute largest portion of C that can be evenly partitioned into CBS blocks
	for(n1 = 0; n1 < (N / n_c); n1++) {
		for(m1 = 0; m1 < (M / (p*m_c)); m1++) {
			// pragma omp here (i_c loop)
			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p; m++) {
				for(k1 = 0; k1 < (K / k_c); k1++) {
					// pragma omp also here possible (j_r loop)
					// #pragma omp parallel num_threads(p)
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
												
							// bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
					  //  		m_r, n_r, k_c, &alpha, 
					  //  		&A_p[m1*p*(K/k_c) + k1*p + m ][m_reg*m_r*k_c], 
					  //  		rsa, csa, 
					  //  		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], 
					  //  		rsb, csb, &beta, 
					  //  		&C_p[m + m1*p + n1*(M/m_c)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					  //  		rsc, csc);

							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				int k_c1 = K % k_c;
				if(k_c1) {
					int k1_n = K / k_c;
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}
			}
		}

		// compute the final row of CBS result blocks each of shape p_l*m_c x n_c
		if((M % (p*m_c))) {

			p_l = (int) ceil(((double) (M % (p*m_c))) / m_c);
			m1 = (M / (p*m_c));
			// p_l = 2, m1 = 1

			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p_l; m++) {
				for(k1 = 0; k1 < (K / k_c); k1++) {
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							// printf("gtrudfbgiu. %d \n", (K/k_c + k_pad)  * (M/m_c + m_pad)  );
												
							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p_l + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				int k1_n = K / k_c;
				int k_c1 = K % k_c;

				if(k_c1) {
				
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p_l + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}
			}
		}
	}









	// compute last column of CBS result blocks of size m_c x n_c1
	n1 = (N / n_c);
	p_ln = (int) ceil(((double) (N % n_c)) / (x));
	n_c1 = x * p_ln;
	printf("here %d \n", n_c1);
	if(n_c1) {	
		printf("gotttt\n");
		for(m1 = 0; m1 < (M / (p*m_c)); m1++) {
			// pragma omp here (i_c loop)
			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p; m++) {
				for(k1 = 0; k1 < (K / k_c); k1++) {
					// pragma omp also here possible (j_r loop)
					// #pragma omp parallel num_threads(p)
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
												
							// bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
					  //  		m_r, n_r, k_c, &alpha, 
					  //  		&A_p[m1*p*(K/k_c) + k1*p + m ][m_reg*m_r*k_c], 
					  //  		rsa, csa, 
					  //  		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], 
					  //  		rsb, csb, &beta, 
					  //  		&C_p[m + m1*p + n1*(M/m_c)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					  //  		rsc, csc);

							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c1 + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				int k_c1 = K % k_c;
				if(k_c1) {
					int k1_n = K / k_c;
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c1 + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}
			}
		}


		// compute the final CBS result block of shape p_l*m_c x n_c1
		if((M % (p*m_c))) {

			p_l = (int) ceil(((double) (M % (p*m_c))) / m_c);
			m1 = (M / (p*m_c));

			// p_l = 2, m1 = 1

			#pragma omp parallel for private(n_reg,m_reg,m,k1)
			for(m = 0; m < p_l; m++) {
				for(k1 = 0; k1 < (K / k_c); k1++) {
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							// printf("gtrudfbgiu. %d \n", (K/k_c + k_pad)  * (M/m_c + m_pad)  );
												
							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p_l + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c1 + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				int k1_n = K / k_c;
				int k_c1 = K % k_c;
				
				if(k_c1) {
				
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p_l + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c1 + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*(M/m_c + m_pad)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}


				}
			}
		}















	}
	
	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	printf("GEMM time: %f \n", diff_t); 


	printf("packed C\n\n\n");
	// print_packed_C(C_p, M, N, m_c, n_c);
	printf("\n\n\n\n");
	// exit(1);

	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p); 
	unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 
	// for(int i = 0; i < M*N; i++) {
	// 	printf("%f ", C[i]);
	// }
	// printf("\n\n\n\n\n");


	for(int i = 0; i < (K/k_c + k_pad)  * (M/m_c + m_pad); i++) {
		free(A_p[i]);
	}

	for(int i = 0; i < (M/m_c + m_pad) * (N/n_c + n_pad); i++) {
		free(C_p[i]);
	}

	free(A_p);
	free(B_p);
	free(C_p);
}



void pack_A(double* A, double** A_p, int M, int K, int m_c, int k_c, int m_r, int p) {

	int ind1 = 0;
	int ind2 = 0;
	int ret;
	int m_pad = (p*m_c) - (M % (p*m_c)); 
	int m1;
	int k1;
	int k_c1 = (K % k_c);
	int p_l;

	int m2;

	int M_rem;

	// main portion of A that evenly fits into CBS blocks with p m_cxk_c blocks
	for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				ind2 = 0;
				ret = posix_memalign((void**) &A_p[ind1], 64, k_c * m_c * sizeof(double));
				
				if(ret) {
					printf("posix memalign error\n");
					exit(1);
				}

				for(int m3 = 0; m3 < m_c; m3 += m_r) {
					for(int i = 0; i < k_c; i++) {
						for(int j = 0; j < m_r; j++) {
							A_p[ind1][ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
							ind2++;
						}
					}
				}

				ind1++;
			}
		}

		if(k_c1) {
			k1 = K - (K%k_c);
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				ind2 = 0;
				ret = posix_memalign((void**) &A_p[ind1], 64, k_c1 * m_c * sizeof(double));			
				if(ret) {
					printf("posix memalign error\n");
					exit(1);
				}

				for(int m3 = 0; m3 < m_c; m3 += m_r) {
					for(int i = 0; i < k_c1; i++) {
						for(int j = 0; j < m_r; j++) {
							A_p[ind1][ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
							ind2++;
						}
					}
				}

				ind1++;
			}
		}
	}

	// Process the final row of CBS blocks (with p_l  m_c x k_c blocks) and perform M-dim padding
	if(M % (p*m_c)) {	

		p_l = (int) ceil(((double) (M % (p*m_c))) / m_c);
		m1 = (M - (M % (p*m_c)));
		M_rem = m1 + p_l*m_c;

		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			for(m2 = 0; m2 < p_l*m_c; m2 += m_c) {

				ind2 = 0;
				ret = posix_memalign((void**) &A_p[ind1], 64, k_c * m_c * sizeof(double));
				
				if(ret) {
					printf("posix memalign error\n");
					exit(1);
				}

				for(int m3 = 0; m3 < m_c; m3 += m_r) {
					for(int i = 0; i < k_c; i++) {
						for(int j = 0; j < m_r; j++) {

							if((m1+m2+m3+j) >=  M) {
								A_p[ind1][ind2] = 0.0;
							} else {
								A_p[ind1][ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
							}

							ind2++;
						}
					}
				}

				ind1++;
			}
		}
		
		// Final CBS block (with p_l m_c x k_c1 blocks) present in the lower right hand corner of A 
		if(k_c1) {

			k1 = K - (K%k_c);
			for(m2 = 0; m2 < p_l*m_c; m2 += m_c) {
				ind2 = 0;
				ret = posix_memalign((void**) &A_p[ind1], 64, k_c1 * m_c * sizeof(double));			
				if(ret) {
					printf("posix memalign error\n");
					exit(1);
				}

				for(int m3 = 0; m3 < m_c; m3 += m_r) {
					for(int i = 0; i < k_c1; i++) {
						for(int j = 0; j < m_r; j++) {

							if((m1 + m2 + m3 + j) >=  M) {
								A_p[ind1][ind2] = 0.0;
							} else {
								A_p[ind1][ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
							}

							ind2++;
						}
					}
				}

				ind1++;
			}
		}
	}
}



void pack_B(double* B, double* B_p, int K, int N, int k_c, int n_c, int n_r, int alpha_n, int m_c) {

	int x = (int) (alpha_n * m_c);
	int k1, k_c1, n1, n_c1, p_l;
	int ind1 = 0;

	// main portion of B that evenly fits into CBS blocks of size k_c x n_c 
	for(n1 = 0; n1 < (N - (N%n_c)); n1 += n_c) {
		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			for(int n2 = 0; n2 < n_c; n2 += n_r) {
				for(int i = 0; i < k_c; i++) {
					for(int j = 0; j < n_r; j++) {
						B_p[ind1] = B[n1 + k1*N + n2 + i*N + j];
						ind1++;
					}
				}
			}
		}

		k1 = (K - (K%k_c));
		k_c1 = (K % k_c);
		if(k_c1) {
			for(int n2 = 0; n2 < n_c; n2 += n_r) {
				for(int i = 0; i < k_c1; i++) {
					for(int j = 0; j < n_r; j++) {
						B_p[ind1] = B[n1 + k1*N + n2 + i*N + j];
						ind1++;
					}
				}
			}
		}
	}

	// Process the final column of CBS blocks (sized k_c x n_c1) and perform N-dim padding 
	n1 = (N - (N%n_c));
	p_l = (int) ceil(((double) (N % n_c)) / x);
	n_c1 = x * p_l;

	if(n_c1) {	

		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			for(int n2 = 0; n2 < n_c1; n2 += n_r) {
				for(int i = 0; i < k_c; i++) {
					for(int j = 0; j < n_r; j++) {

						if((n1 + n2 + j) >=  N) {
							B_p[ind1] = 0.0;
						} else {
							B_p[ind1] = B[n1 + k1*N + n2 + i*N + j];
						}

						ind1++;
					}
				}
			}
		}

		// Final CBS block (with k_c1 x n_c1 blocks) present in the lower right hand corner of B 
		k1 = (K - (K%k_c));
		k_c1 = (K % k_c);
		if(k_c1) {
			for(int n2 = 0; n2 < n_c1; n2 += n_r) {
				for(int i = 0; i < k_c1; i++) {
					for(int j = 0; j < n_r; j++) {

						if((n1 + n2 + j) >=  N) {
							B_p[ind1] = 0.0;
						} else {
							B_p[ind1] = B[n1 + k1*N + n2 + i*N + j];
						}

						ind1++;
					}
				}
			}
		}
	}
}


void pack_C(double** C_p, int M, int N, int m_c, int n_c, int p, int alpha_n) {

	int ind1 = 0;
	int ret;

	int m, n_c1, p_lm, p_ln;
	int x = (int) (alpha_n * m_c);

	// p_lm = (M % (p*m_c)) / m_c;
	// m1 = (M - (M % (p*m_c)));
	// M_rem = m1 + p_lm*m_c;

	for(int n = 0; n < (N/n_c); n++) {
		for(m = 0; m < (M/(p*m_c)); m++) {
			for(int p1 = 0; p1 < p; p1++) {

				ret = posix_memalign((void**) &C_p[ind1], 64, m_c * n_c * sizeof(double));
				if(ret) {
					printf("posix memalign error\n");
					exit(1);
				}

				ind1++; 
			}
		}

		// last row of CBS result blocks (with p_lm instead of p)
		p_lm = (int) ceil(((double) (M % (p*m_c))) / m_c);
		for(int p1 = 0; p1 < p_lm; p1++) {
			ret = posix_memalign((void**) &C_p[ind1], 64, m_c * n_c * sizeof(double));
			if(ret) {
				printf("posix memalign error\n");
				exit(1);
			}
			ind1++; 
		}
	}

	// Process the final column of CBS blocks (with m_c x n_c1 blocks)  
	p_ln = (int) ceil(((double) (N % n_c)) / (x));
	n_c1 = x * p_ln ;

	if(n_c1) {	

		for(m = 0; m < (M/(p*m_c)); m++) {
			for(int p1 = 0; p1 < p; p1++) {

				ret = posix_memalign((void**) &C_p[ind1], 64, m_c * n_c1 * sizeof(double));
				if(ret) {
					printf("posix memalign error\n");
					exit(1);
				}

				ind1++; 
			}
		}

		// Process the final CBS block (with size m_c x n_c1) in lower right corner
		p_lm = (int) ceil(((double) (M % (p*m_c))) / m_c);
		for(int p1 = 0; p1 < p_lm; p1++) {
			ret = posix_memalign((void**) &C_p[ind1], 64, m_c * n_c1 * sizeof(double));
			if(ret) {
				printf("posix memalign error\n");
				exit(1);
			}
			ind1++; 
		}
	}
}


void unpack_C_rsc(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n) {

	int m_pad = (M % m_c) ? 1 : 0; 
	int x = (int) (alpha_n * m_c);

	int m1, p_lm, p_ln, n1, n_c1;
	int ind1 = 0;
	for(m1 = 0; m1 < (M/(p*m_c)); m1++) {
		for(int m = 0; m < p; m++) {
			for(int m2 = 0; m2 < m_c/m_r; m2++) {
				for(int m3 = 0; m3 < m_r; m3++) { 
					for(n1 = 0; n1 < (N/n_c); n1++) {
						for(int i = 0; i < n_c/n_r; i++) {
							for(int j = 0; j < n_r; j++) {
								C[ind1] = C_p[m1*p + m + n1*(M/m_c + m_pad)][m2*m_r*n_r + m3*n_r + i*m_c*n_r + j];
								ind1++;
							}
						}
					}

					n1 = N/n_c;
					p_ln = (int) ceil(((double) (N % n_c)) / x);
					n_c1 = x * p_ln;

					for(int i = 0; i < n_c1/n_r; i++) {
						for(int j = 0; j < n_r; j++) {
							if( (i*n_r + j + n1*n_c) < N) {
								C[ind1] = C_p[m1*p + m + n1*(M/m_c + m_pad)][m2*m_r*n_r + m3*n_r + i*m_c*n_r + j];
								ind1++;
							}
						}
					}
				}
			}
		}
	}

	m1 = M / (p*m_c);
	p_lm = (int) ceil(((double) (M % (p*m_c))) / m_c);

	for(int m = 0; m < p_lm; m++) {
		for(int m2 = 0; m2 < m_c/m_r; m2++) {
			for(int m3 = 0; m3 < m_r; m3++) { 
				for(n1 = 0; n1 < (N/n_c); n1++) {
					for(int i = 0; i < n_c/n_r; i++) {
						for(int j = 0; j < n_r; j++) {
							// ignore zero-padded rows
							if( (m3 + m2*m_r + m*m_c + m1*p*m_c) < M) {
								C[ind1] = C_p[m1*p + m + n1*(M/m_c + m_pad)][m2*m_r*n_r + m3*n_r + i*m_c*n_r + j];
								ind1++;
							}
						}
					}
				}


				n1 = N/n_c;
				p_ln = (int) ceil(((double) (N % n_c)) / x);
				n_c1 = x * p_ln;

				for(int i = 0; i < n_c1/n_r; i++) {
					for(int j = 0; j < n_r; j++) {
						if( (i*n_r + j + n1*n_c) < N  && (m3 + m2*m_r + m*m_c + m1*p*m_c) < M) {
							C[ind1] = C_p[m1*p + m + n1*(M/m_c + m_pad)][m2*m_r*n_r + m3*n_r + i*m_c*n_r + j];
							ind1++;
						}
					}
				}
			}
		}
	}
	
	printf("unpack cnt = %d\n", ind1);

}


void unpack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p) {
	printf("etohglnkoibjn\n");


	int m_pad = (M % m_c) ? 1 : 0; 
	int m1, p_l;

	int ind1 = 0;
	for(int n3 = 0; n3 < (N/n_c); n3++) {
		for(int n2 = 0; n2 < (n_c/n_r); n2++) {
			for(int n1 = 0; n1 < n_r; n1++) {
				for(m1 = 0; m1 < (M/(p*m_c)); m1++) {
					for(int m = 0; m < p; m++) {
						for(int i = 0; i < (m_c/m_r); i++) {
							for(int j = 0; j< m_r; j++) {

								printf("ind1 %d \n", n3*(M/m_c + m_pad)  + m1*p + m);
								printf("ind2 %d \n", n2*m_c*n_r + n1*m_r + i*m_r*n_r + j);
								
								printf("dude\n");
								C[ind1] = C_p[n3*(M/m_c + m_pad)  + m1*p + m][n2*m_c*n_r + n1*m_r + i*m_r*n_r + j];
								printf("done\n");
								ind1++;
							}
						}
					}
				}

				m1 = M / (p*m_c);
				p_l = (int) ceil(((double) (M % (p*m_c))) / m_c);


				for(int m = 0; m < p_l; m++) {
					for(int i = 0; i < (m_c/m_r); i++) {
						for(int j = 0; j< m_r; j++) {
							C[ind1] = C_p[n3*(M/m_c + m_pad)  + m1*p + m][n2*m_c*n_r + n1*m_r + i*m_r*n_r + j];
							ind1++;
						}
					}
				}
			}
		}
	}
}


void cake_dgemm_checker(double* A, double* B, double* C, int N, int M, int K) {

	double* C_check = calloc(M * N , sizeof( double ));

	#pragma omp parallel for
	for(int m = 0; m < M; m++) {
		for(int n = 0; n < N; n++) {
			C_check[m*N + n] = 0.0;
			for(int k = 0; k < K; k++) {
				C_check[m*N + n] += A[m*K + k] * B[k*N + n];
			}
			// printf("%f ", C_check[m*N + n]);
		}
		// printf("\n");
	}
	// printf("\n\n\n\n\n");
	// exit(1);

    int CORRECT = 1;
    int cnt = 0;
    int ind1 = 0;
    double eps = 1e-11; // machine precision level

	for(int m = 0; m < M; m++) {
	    for(int n = 0; n < N; n++) {
	        // if(C_check[m1*N + n1] != C[ind1]) {
	        if(fabs(C_check[ind1] - C[ind1]) > eps) {
	            cnt++;
	            CORRECT = 0;
	        }

        // printf("%f\t%f\n", C_check[ind1], C[ind1]);

        ind1++; 
      }
    }

    printf("\n\n");

	if(CORRECT) {
		printf("CORRECT!\n");
	} else {
		printf("WRONG!\n");
		printf("%d\n", cnt);
	}

	free(C_check);

	// for (int n1 = 0; n1 < N; n1++) {
 //      for (int m1 = 0; m1 < M; m1++) {
 //        printf("%f ", C_check[m1*N + n1]);
 //      }
 //    }
	// printf("\n\n\n\n");
}



void print_packed_A(double** A_p, int M, int K, int m_c, int k_c, int p) {

	int k_pad = (K % k_c) ? 1 : 0; 
	int m_pad = (M % m_c) ? 1 : 0; 

	int ind = 0;
	for(int i = 0; i < ((K/k_c + k_pad) * (M/m_c + m_pad)); i++) {


		if(i!=0 && (i % (K/k_c * p)) == 0) { 
			for(int j = 0; j < m_c*(K % k_c); j++) {
				printf("%f ", A_p[i][j]);
			}
			ind++;

		// if(( (i+1) % (K/k_c + k_pad)) == 0 ) { 
		// 	for(int j = 0; j < m_c*(K % k_c); j++) {
		// 		printf("%f ", A_p[i][j]);
		// 	}
		} else {
			for(int j = 0; j < m_c*k_c; j++) {
				printf("%f ", A_p[i][j]);
			}	
		}	

		printf("\n\n");
	}
}


void print_packed_C(double** C_p, int M, int N, int m_c, int n_c) {

	int m_pad = (M % m_c) ? 1 : 0; 

	for(int i = 0; i < ((N/n_c) * (M/m_c + m_pad)); i++) {
		for(int j = 0; j < m_c*n_c; j++) {
			printf("%f ", C_p[i][j]);
		}	

		printf("\n\n");
	}
}


// find cache size at levels L1d,L1i,L2,and L3 using lscpu
int get_cache_size(char* level) {

	int len, size = 0;
	FILE *fp;
	char ret[16];
	char command[128];

	sprintf(command, "lscpu --caches=NAME,ONE-SIZE \
					| grep %s \
					| grep -Eo '[0-9]*M|[0-9]*K|0-9*G' \
					| tr -d '\n'", level);
	fp = popen(command, "r");

	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit(1);
	}

	if(fgets(ret, sizeof(ret), fp) == NULL) {
		printf("lscpu error\n");
	}

	pclose(fp);

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


int get_block_dim(int m_r, int n_r, double alpha_n) {

	int mc_L2 = 0, mc_L3 = 0;
	// find L3 and L2 cache sizes
	int max_threads = omp_get_max_threads() / 2; // 2-way hyperthreaded
	int L2_size = get_cache_size("L2");
	int L3_size = get_cache_size("L3");
	// solves for the optimal block size m_c and k_c based on the L3 size
	// L3_size >= (alpha_n*p*(1+p)) * x^2     (solve for x = m_c = k_c) 
	// We only use half of the each cache to prevent our working blocks from being evicted
	mc_L3 = (int) sqrt((((double) L3_size) / (sizeof(double) * 2))  
							/ (alpha_n*max_threads*(max_threads+1)));
	mc_L3 -= (mc_L3 % lcm(m_r, n_r));

	// solves for optimal mc,kc based on L2 size
	// L2_size >= x^2     (solve for x = m_c = k_c) 
	mc_L2 = (int) sqrt(((double) L2_size) / (sizeof(double) * 2));
	mc_L2 -= (mc_L2 % lcm(m_r, n_r));

	printf("%d %d %d\n", L2_size,L3_size,max_threads);
	// return min of possible L2 and L3 cache block sizes
	return (mc_L3 < mc_L2 ? mc_L3 : mc_L2);
}


// randomized double precision matrices in range [-1,1]
void rand_init(double* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (double) rand() / RAND_MAX*2.0 - 1.0;
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
