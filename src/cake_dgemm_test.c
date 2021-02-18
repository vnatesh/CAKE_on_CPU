#include <stdio.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include "blis.h"
 
void pack_B(double* B, double* B_p, int K, int N, int k_c, int n_c, int n_r, int alpha_n, int m_c);
void pack_A(double* A, double** A_p, int M, int K, int m_c, int k_c, int m_r, int p);
void pack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n);
void unpack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);
void rand_init(double* mat, int r, int c);
void cake_dgemm_checker(double* A, double* B, double* C, int N, int M, int K);
void cake_dgemm(double* A, double* B, double beta_user, double* C, int M, int N, int K, int p);

int get_block_dim(int m_r, int n_r, double alpha_n);
int get_cache_size(char* level);
int lcm(int n1, int n2);

void unpack_C_rsc(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);

void print_array(double* arr, int len);

void set_ob_A(double* A, double* A_p, int M, int K, int m1, int k1, 
				int m2, int m_c, int k_c, int m_r, bool pad);
void set_ob_C(double* C, double* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad);

static bool DEBUG = 0;

int main( int argc, char** argv ) {


	struct timeval start, end;
	double diff_t;

    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

	int M, K, N, p;

	// M = 1111;
	// K = 1111;
	// N = 2880;
	M = 2111;
	K = 2111;
	N = 2111;


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



	gettimeofday (&start, NULL);

	// initialize A and B
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);
	// rand_init(C, M, N);
	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	printf("init time: %f \n", diff_t); 


    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         m, n, k, alpha, A, k, B, n, beta, C, n);

	double beta = 1.0;
	cake_dgemm(A, B, beta, C, M, N, K, p);
	// bli_dprintm( "C: ", M, N, C, N, 1, "%4.4f", "" );
	cake_dgemm_checker(A, B, C, N, M, K);

	free(A);
	free(B);
	free(C);

	return 0;
}



void cake_dgemm(double* A, double* B, double beta_user, double* C, int M, int N, int K, int p) {
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
	double** A_p; 
	double** C_p;
	double* B_p;

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

	#pragma omp parallel sections
	{
	    #pragma omp section
	    {
	    // pack A
			A_p = (double**) malloc( (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l) * sizeof( double* ));
			pack_A(A, A_p, M, K, m_c, k_c, m_r, p);
			// exit(1);
	    }
	    #pragma omp section
	    {
			// pack B
			if(posix_memalign((void**) &B_p, 64, K * N_b * sizeof(double))) {
				printf("posix memalign error\n");
				exit(1);
			}
			pack_B(B, B_p, K, N, k_c, n_c, n_r, alpha_n, m_c);

	    }

	    #pragma omp section
	    {
			C_p = (double**) malloc((((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad) * sizeof( double* ));
			if(beta_user > 0)  pack_C(C, C_p, M, N, m_c, n_c, m_r, n_r, p, alpha_n);
	    }

	}

		// create_C();

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	printf("packing time: %f \n", diff_t); 



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
	int n_reg, m_reg, m, k1, m1, n1;

	int m_c1 = mr_per_core * m_r;
	int m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;

	int k1_n = K / k_c;
	int k_c1 = K % k_c;

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
							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				if(k_c1) {
					int k1_n = K / k_c;
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
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
												
							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p_l + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				if(k_c1) {
				
					for(n_reg = 0; n_reg < (n_c / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_cx / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p_l + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
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
												
							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c1 + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}

				if(k_c1) {
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c1 + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_c*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
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
												
							bli_dgemm_haswell_asm_6x8(k_c, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1*p_l + m ][m_reg*m_r*k_c], 
					   		&B_p[n1*K*n_c + k1*k_c*n_c1 + n_reg*k_c*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, NULL);
						}
					}
				}
				
				if(k_c1) {
				
					for(n_reg = 0; n_reg < (n_c1 / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_cx / m_r); m_reg++) {
							bli_dgemm_haswell_asm_6x8( k_c1, &alpha, 
					   		&A_p[m1*p*(K/k_c + k_pad) + k1_n*p_l + m ][m_reg*m_r*k_c1], 
					   		&B_p[n1*K*n_c + k1_n*k_c*n_c1 + n_reg*k_c1*n_r], &beta, 
					   		&C_p[m + m1*p + n1*((M / (p*m_c))*p + p_l)][n_reg*m_cx*n_r + m_reg*m_r*n_r], 
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
// exit(1);
	// print_packed_C(C_p, M, N, m_c, n_c);
	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p);

	gettimeofday (&start, NULL);

	unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 

	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	printf("unpacking time: %f \n", diff_t); 

	for(int i = 0; i < (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l); i++) {
		free(A_p[i]);
	}

	for(int i = 0; i < (((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad); i++) {
		free(C_p[i]);
	}

	free(A_p);
	free(B_p);
	free(C_p);
}




void pack_A(double* A, double** A_p, int M, int K, int m_c, int k_c, int m_r, int p) {

	int ind1 = 0;
	int m1, k1, m2, p_l;
	int k_c1 = (K % k_c);
	int mr_rem = (int) ceil( ((double) (M % (p*m_c))) / m_r) ;
	int mr_per_core = (int) ceil( ((double) mr_rem) / p );
	int m_c1 = mr_per_core * m_r;

	if(mr_per_core) 
		p_l = (int) ceil( ((double) mr_rem) / mr_per_core);
	else
		p_l = 0;

	int m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;

	// main portion of A that evenly fits into CBS blocks each with p m_cxk_c OBs
	for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {
				
				if(posix_memalign((void**) &A_p[ind1], 64, k_c * m_c * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_A(A, A_p[ind1], M, K, m1, k1, m2, m_c, k_c, m_r, 0);

				if(DEBUG) print_array(A_p[ind1], k_c * m_c);
				ind1++;
			}
		}
		// right-most column of CBS blocks each with p m_c x k_c1 OBs
		if(k_c1) {
			k1 = K - (K%k_c);
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				if(posix_memalign((void**) &A_p[ind1], 64, k_c1 * m_c * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_A(A, A_p[ind1], M, K, m1, k1, m2, m_c, k_c1, m_r, 0);
				if(DEBUG) print_array(A_p[ind1], k_c1 * m_c);
				ind1++;
			}
		}
	}

	// Process bottom-most rows of CBS blocks and perform M-dim padding
	if(M % (p*m_c)) {	

		m1 = (M - (M % (p*m_c)));

		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {
				
				if(posix_memalign((void**) &A_p[ind1], 64, k_c * m_c1 * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_A(A, A_p[ind1], M, K, m1, k1, m2, m_c1, k_c, m_r, 0);
				if(DEBUG) print_array(A_p[ind1], k_c * m_c1);
				ind1++;
			}

			// final row of CBS blocks each with m_c1_last_core x k_c
			m2 = (p_l-1) * m_c1;
			if(posix_memalign((void**) &A_p[ind1], 64, k_c * m_c1_last_core * sizeof(double))) {
				printf("posix memalign error\n");
				exit(1);
			}

			set_ob_A(A, A_p[ind1], M, K, m1, k1, m2, m_c1_last_core, k_c, m_r, 1);
			if(DEBUG) print_array(A_p[ind1], k_c * m_c1_last_core);
			ind1++;
		}

		// Final CBS block (with p_l-1 m_c1 x k_c1 OBs and 1 m_c1_last_core x k_c1 OB) 
		// present in the lower right hand corner of A 
		if(k_c1) {

			k1 = K - (K%k_c);
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {

				if(posix_memalign((void**) &A_p[ind1], 64, k_c1 * m_c1 * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_A(A, A_p[ind1], M, K, m1, k1, m2, m_c1, k_c1, m_r, 0);
				if(DEBUG) print_array(A_p[ind1], k_c1 * m_c1);
				ind1++;
			}

			// last OB of A has shape m_c1_last_core x k_c1 
			m2 = (p_l-1) * m_c1;
			
			if(posix_memalign((void**) &A_p[ind1], 64, k_c1 * m_c1_last_core * sizeof(double))) {
				printf("posix memalign error\n");
				exit(1);
			}

			set_ob_A(A, A_p[ind1], M, K, m1, k1, m2, m_c1_last_core, k_c1, m_r, 1);
			if(DEBUG) print_array(A_p[ind1], k_c1 * m_c1_last_core);
			ind1++;
		}
	}
}



void pack_B(double* B, double* B_p, int K, int N, int k_c, int n_c, int n_r, int alpha_n, int m_c) {

	int k1, k_c1, n1, n_c1, nr_rem;
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
	nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
	n_c1 = nr_rem * n_r;

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


// initialize an operation block of matrix A
void set_ob_A(double* A, double* A_p, int M, int K, int m1, int k1, int m2, int m_c, int k_c, int m_r, bool pad) {

	int	ind2 = 0;
	
	if(pad) {
		for(int m3 = 0; m3 < m_c; m3 += m_r) {
			for(int i = 0; i < k_c; i++) {
				for(int j = 0; j < m_r; j++) {

					if((m1 + m2 + m3 + j) >=  M) {
						A_p[ind2] = 0.0;
					} else {
						A_p[ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
					}

					ind2++;
				}
			}
		}		
	} 

	else {
		for(int m3 = 0; m3 < m_c; m3 += m_r) {
			for(int i = 0; i < k_c; i++) {
				for(int j = 0; j < m_r; j++) {
					A_p[ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
					ind2++;
				}
			}
		}
	}
}



void set_ob_C(double* C, double* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad) {

	int	ind2 = 0;

	if(pad) {

		for(int n2 = 0; n2 < n_c; n2 += n_r) {
			for(int m3 = 0; m3 < m_c; m3 += m_r) {
				for(int i = 0; i < m_r; i++) {
					for(int j = 0; j < n_r; j++) {
						if((n1 + n2 + j) >= N  ||  (m1 + m2 + m3 + i) >=  M) {
							C_p[ind2] = 0.0;
						} else {
							C_p[ind2] = C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j];
						}
						ind2++;
					}
				}
			}
		}

	} else {

		for(int n2 = 0; n2 < n_c; n2 += n_r) {
			for(int m3 = 0; m3 < m_c; m3 += m_r) {
				for(int i = 0; i < m_r; i++) {
					for(int j = 0; j < n_r; j++) {
						C_p[ind2] = C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j];
						ind2++;
					}
				}
			}
		}
	}
}


void pack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n) {

	int n1, m1, m2, p_l;

	int mr_rem = (int) ceil( ((double) (M % (p*m_c))) / m_r) ;
	int mr_per_core = (int) ceil( ((double) mr_rem) / p );
	int m_c1 = mr_per_core * m_r;

	if(mr_per_core) 
		p_l = (int) ceil( ((double) mr_rem) / mr_per_core);
	else
		p_l = 0;


	int m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;

	int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
	int n_c1 = nr_rem * n_r;

	int ind1 = 0;

	// main portion of C that evenly fits into CBS blocks each with p m_cxn_c OBs
	for(n1 = 0; n1 < (N - (N%n_c)); n1 += n_c) {
		for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				if(posix_memalign((void**) &C_p[ind1], 64, m_c * n_c * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_C(C, C_p[ind1], M, N, m1, n1, m2, m_c, n_c, m_r, n_r, 0);
				if(DEBUG) print_array(C_p[ind1], m_c * n_c);
				ind1++;
			}
		}

		// bottom row of CBS blocks with p_l-1 OBs of m_c1 x n_c and 1 OBs of shape m_c1_last_core x n_c
		if(M % (p*m_c)) {	

			m1 = (M - (M % (p*m_c)));

			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {

				if(posix_memalign((void**) &C_p[ind1], 64, m_c1 * n_c * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_C(C, C_p[ind1], M, N, m1, n1, m2, m_c1, n_c, m_r, n_r, 0);
				if(DEBUG) print_array(C_p[ind1], m_c1 * n_c);
				ind1++;
			}

			m2 = (p_l-1) * m_c1;
			if(posix_memalign((void**) &C_p[ind1], 64, m_c1_last_core * n_c * sizeof(double))) {
				printf("posix memalign error\n");
				exit(1);
			}

			set_ob_C(C, C_p[ind1], M, N, m1, n1, m2, m_c1_last_core, n_c, m_r, n_r, 1);
			if(DEBUG) print_array(C_p[ind1], m_c1 * n_c);
			ind1++;
		}
	}

	// right-most column of CBS blocks with p OBs of shape m_c x n_c1
	n1 = (N - (N%n_c));

	if(n_c1) {	

		for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				if(posix_memalign((void**) &C_p[ind1], 64, m_c * n_c1 * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_C(C, C_p[ind1], M, N, m1, n1, m2, m_c, n_c1, m_r, n_r, 1);
				if(DEBUG) print_array(C_p[ind1], m_c * n_c1);
				ind1++;
			}
		}

		if(M % (p*m_c)) {	

			m1 = (M - (M % (p*m_c)));

			// last row of CBS blocks with p_l-1 m_c1 x n_c1 OBs and 1 m_c1_last_core x n_c1 OB
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {

				if(posix_memalign((void**) &C_p[ind1], 64, m_c1 * n_c1 * sizeof(double))) {
					printf("posix memalign error\n");
					exit(1);
				}

				set_ob_C(C, C_p[ind1], M, N, m1, n1, m2, m_c1, n_c1, m_r, n_r, 1);
				if(DEBUG) print_array(C_p[ind1], m_c1 * n_c1);
				ind1++;
			}

			// last OB in C (lower right corner) with shape m_c1_last_core * n_c1
			m2 = (p_l-1) * m_c1;
			if(posix_memalign((void**) &C_p[ind1], 64, m_c1_last_core * n_c1 * sizeof(double))) {
				printf("posix memalign error\n");
				exit(1);
			}

			set_ob_C(C, C_p[ind1], M, N, m1, n1, m2, m_c1_last_core, n_c1, m_r, n_r, 1);
			if(DEBUG) print_array(C_p[ind1], m_c1_last_core * n_c1);
			ind1++;
		}
	}
}



void unpack_C_rsc(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n) {

	int m, m1, n1, p_l;
	int ind1 = 0;

	int mr_rem = (int) ceil( ((double) (M % (p*m_c))) / m_r) ;
	int mr_per_core = (int) ceil( ((double) mr_rem) / p );
	int m_c1 = mr_per_core * m_r;

	if(mr_per_core) 
		p_l = (int) ceil( ((double) mr_rem) / mr_per_core);
	else
		p_l = 0;

	int m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;

	int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
	int n_c1 = nr_rem * n_r;


	for(m1 = 0; m1 < (M/(p*m_c)); m1++) {
		for(int m = 0; m < p; m++) {
			for(int m2 = 0; m2 < m_c/m_r; m2++) {
				for(int m3 = 0; m3 < m_r; m3++) { 
					for(n1 = 0; n1 < (N/n_c); n1++) {
						for(int i = 0; i < n_c/n_r; i++) {
							for(int j = 0; j < n_r; j++) {
								C[ind1] = C_p[m1*p + m + n1*((M/(p*m_c))*p  + p_l)][m2*m_r*n_r + m3*n_r + i*m_c*n_r + j];
								ind1++;
							}
						}
					}

					n1 = N/n_c;

					for(int i = 0; i < n_c1/n_r; i++) {
						for(int j = 0; j < n_r; j++) {
							if( (i*n_r + j + n1*n_c) < N) {
								C[ind1] = C_p[m1*p + m + n1*((M/(p*m_c))*p  + p_l)][m2*m_r*n_r + m3*n_r + i*m_c*n_r + j];
								ind1++;
							}
						}
					}
				}
			}
		}	
	}

	m1 = M / (p*m_c);

	for(m = 0; m < p_l-1; m++) {

		for(int m2 = 0; m2 < m_c1/m_r; m2++) {
			for(int m3 = 0; m3 < m_r; m3++) { 
				for(n1 = 0; n1 < (N/n_c); n1++) {
					for(int i = 0; i < n_c/n_r; i++) {
						for(int j = 0; j < n_r; j++) {
							C[ind1] = C_p[m1*p + m + n1*((M/(p*m_c))*p + p_l)][m2*m_r*n_r + m3*n_r + i*m_c1*n_r + j];
							ind1++;		
						}
					}
				}


				n1 = N/n_c;

				for(int i = 0; i < n_c1/n_r; i++) {
					for(int j = 0; j < n_r; j++) {
						// ignore zero-padded rows
						if( (i*n_r + j + n1*n_c) < N ) {
							C[ind1] = C_p[m1*p + m + n1*((M/(p*m_c))*p + p_l)][m2*m_r*n_r + m3*n_r + i*m_c1*n_r + j];
							ind1++;
						}
					}
				}
			}
		}
	}

	m = p_l - 1;

	for(int m2 = 0; m2 < m_c1_last_core/m_r; m2++) {
		for(int m3 = 0; m3 < m_r; m3++) { 
			for(n1 = 0; n1 < (N/n_c); n1++) {
				for(int i = 0; i < n_c/n_r; i++) {
					for(int j = 0; j < n_r; j++) {
						if( (m3 + m2*m_r + m*m_c1 + m1*p*m_c) < M ) {
							C[ind1] = C_p[m1*p + m + n1*((M/(p*m_c))*p + p_l)][m2*m_r*n_r + m3*n_r + i*m_c1_last_core*n_r + j];
							ind1++;		
						}
					}
				}
			}


			n1 = N/n_c;

			for(int i = 0; i < n_c1/n_r; i++) {
				for(int j = 0; j < n_r; j++) {
					// ignore zero-padded rows
					if( ((i*n_r + j + n1*n_c) < N)  && ((m3 + m2*m_r + m*m_c1 + m1*p*m_c) < M)) {
						C[ind1] = C_p[m1*p + m + n1*((M/(p*m_c))*p + p_l)][m2*m_r*n_r + m3*n_r + i*m_c1_last_core*n_r + j];
						ind1++;
					}
				}
			}
		}
	}
}


void unpack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p) {

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
			// C_check[m*N + n] = C[m*N + n];
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
	// L3_size >= p*mc*kc + 2*(kc*alpha*p*mc + p*mc*alpha*p*mc)     (solve for x = m_c = k_c) 
	// We only use ~ half of the each cache to prevent our working blocks from being evicted
	mc_L3 = (int) sqrt((((double) L3_size) / (sizeof(double)))  
							/ (max_threads * (1 + 2*alpha_n + 2*alpha_n*max_threads)));
	mc_L3 -= (mc_L3 % lcm(m_r, n_r));

	// solves for optimal mc,kc based on L2 size
	// L2_size >= 2*(mc*kc + kc*nr) + mc*nr     (solve for x = m_c = k_c) 
	int b = 3*n_r;
	mc_L2 = (int)  (-b + sqrt(b*b + 4*2*(((double) L2_size) / (sizeof(double))))) / (2*2)  ;
	mc_L2 -= (mc_L2 % lcm(m_r, n_r));

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


void print_array(double* arr, int len) {

	for(int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n\n");

}
