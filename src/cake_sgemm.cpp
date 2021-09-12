#include "cake.h"


double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, cake_cntx_t* cake_cntx) {
	// contiguous row-storage (i.e. cs_c = 1) or contiguous column-storage (i.e. rs_c = 1). 
	// This preference comes from how the microkernel is most efficiently able to load/store 
	// elements of C11 from/to memory. Most microkernels use vector instructions to access 
	// contiguous columns (or column segments) of C11
	int m_c, k_c, n_c, m_r, n_r;	
	double alpha_n;
	float alpha, beta;
	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t;
	float* A_p; 
	float** C_p;
	float* B_p;
	inc_t rsc, csc;


	double times;

	// inc_t rsa, csa;
	// inc_t rsb, csb;

	clock_gettime(CLOCK_REALTIME, &start);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	m_r = cake_cntx->mr;
	n_r = cake_cntx->nr;
	alpha_n = cake_cntx->alpha;

    if(DEBUG) printf("M = %d, N = %d, K = %d\n", M, N, K);
    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", m_r, n_r);

    m_c = get_block_dim(cake_cntx, M, p);
   
    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("cntx time: %f \n", diff_t); 


	// printf("hey %d\n",m_c );
	// exit(1);
   //m_c = 96;
	k_c = m_c;
    // m_c = 12;
    // k_c = 6;
    n_c = (int) (alpha_n * p * m_c);
	omp_set_num_threads(p);

    if(DEBUG) printf("mc = %d, kc = %d, nc = %d\n", m_c, k_c, n_c);


	int k_pad = (K % k_c) ? 1 : 0; 
	int n_pad = (N % n_c) ? 1 : 0; 
	int m_pad = (M % (p*m_c)) ? 1 : 0; 

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
	clock_gettime(CLOCK_REALTIME, &start);

	// A_p = (float**) malloc( (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l) * sizeof( float* ));
	A_p = (float*) malloc( (m_r*mr_rem + (M /(p*m_c))*p*m_c) * K * sizeof( float ));
	pack_A(A, A_p, M, K, m_c, k_c, m_r, p);
	// exit(1);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("A pack time: %f \n", diff_t ); 



	clock_gettime(CLOCK_REALTIME, &start);

	if(posix_memalign((void**) &B_p, 64, K * N_b * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}
	pack_B(B, B_p, K, N, k_c, n_c, n_r, alpha_n, m_c);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("B pack time: %f \n", diff_t ); 



	clock_gettime(CLOCK_REALTIME, &start);

	C_p = (float**) malloc((((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad) * sizeof( float* ));
	pack_C(C, C_p, M, N, m_c, n_c, m_r, n_r, p, alpha_n);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("C pack time: %f \n", diff_t ); 

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

	// void (*blis_kernel)(dim_t, float*, float*, float*, 
	// 					float*, float*, inc_t, inc_t, 
	// 					auxinfo_t*, cntx_t*);
	// bli_sgemm_ukernel = bli_sgemm_haswell_asm_6x16;


	clock_gettime(CLOCK_REALTIME, &start);

	int m_c1 = mr_per_core * m_r;
	int m_c1_last_core = (mr_per_core - (p_l*mr_per_core - mr_rem)) * m_r;
	int k_c1 = K % k_c;

	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m_cb, n_c_t, p_used, core;

	// number of CB blocks in the M, N, and K dims
	int Mb = (M / (p*m_c)) + m_pad;
	int Nb = (N / n_c) + n_pad;
	int Kb = (K / k_c) + k_pad;


	for(n = 0; n < Nb; n++) {

		if(n % 2) {
			m_start = Mb - 1;
			m_end = -1;
			m_inc = -1;
		} else {
			m_start = 0;
			m_end = Mb;
			m_inc = 1;
		}

		n_c_t = n_c;
		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
		}

		for(m = m_start; m != m_end; m += m_inc) {

			if(n % 2) {
				if(m % 2) {
					k_start = 0;
					k_end = Kb;
					k_inc = 1;
				} else {
					k_start = Kb - 1;
					k_end = -1;
					k_inc = -1;
				}
			} else {
				if(m % 2) {
					k_start = Kb - 1;
					k_end = -1;
					k_inc = -1;
				} else {
					k_start = 0;
					k_end = Kb;
					k_inc = 1;
				}
			}


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
				int m_c_t, m_c_x, k_c_t, n_reg, m_reg;

				if((m == Mb - 1) && m_pad) {
					m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
					m_c_x = m_c1;
				} else {
					m_c_t = m_c;
					m_c_x = m_c; 
				}


				// pragma omp also here possible (j_r loop)
				for(k = k_start; k != k_end; k += k_inc) {
					
					k_c_t = k_c; 
					if((k == Kb - 1) && k_pad) {
						k_c_t = k_c1;
					}

					int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
					int b_ind = n*K*n_c + k*k_c*n_c_t;
					int c_ind = core + m*p + n*((M / (p*m_c))*p + p_l);

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							
							bli_sgemm_ukernel(k_c_t, &alpha, 
					   		&A_p[a_ind + m_reg*m_r*k_c_t], 
					   		&B_p[b_ind + n_reg*k_c_t*n_r], &beta, 
					   		&C_p[c_ind][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
					   		rsc, csc, NULL, cake_cntx->blis_cntx);
						}
					}
				}
			}
		}
	}



    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);
	// print_packed_C(C_p, M, N, m_c, n_c);
	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p);
	times = diff_t;

	clock_gettime(CLOCK_REALTIME, &start);

	unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);

// cake_sgemm_checker(A, B, C, N, M, K);

	// for(int i = 0; i < (K/k_c + k_pad) * (((M / (p*m_c))*p) + p_l); i++) {
	// 	free(A_p[i]);
	// }

	for(int i = 0; i < (((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad); i++) {
		free(C_p[i]);
	}

	free(A_p);
	free(B_p);
	free(C_p);

	return times;
}


