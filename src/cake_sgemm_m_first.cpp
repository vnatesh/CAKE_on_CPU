#include "cake.h"



double cake_sgemm_m_first(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA , bool packedB, float alpha, float beta) {
	// contiguous row-storage (i.e. cs_c = 1) or contiguous column-storage (i.e. rs_c = 1). 
	// This preference comes from how the microkernel is most efficiently able to load/store 
	// elements of C11 from/to memory. Most microkernels use vector instructions to access 
	// contiguous columns (or column segments) of C11

	int A_sz, B_sz, C_sz;	
	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = init_block_dims(M, N, K, p, cake_cntx, MKN);
	omp_set_num_threads(p);

    if(DEBUG) printf("M = %d, N = %d, K = %d\n", M, N, K);
    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d\n", x->m_c, x->k_c, x->n_c);


	if(packedA) {
		A_p = A;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, 1, x, cake_cntx);
		if(posix_memalign((void**) &A_p, 64, A_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}
		// A_sz = cake_sgemm_packed_A_size(M, K, 1, cake_cntx) / sizeof(float);
	 //    A_p = (float*) calloc(A_sz, sizeof(float));
		pack_A_single_buf_m_first(A, A_p, M, K, p, x, cake_cntx);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A pack time: %f \n", diff_t ); 
	}


	if(packedB) {
		B_p = B;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

	    B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}
	    // B_sz = cake_sgemm_packed_B_size(K, N, p, cake_cntx) / sizeof(float);
	    // B_p = (float*) calloc(B_sz, sizeof(float));
		pack_B_m_first(B, B_p, K, N, p, x, cake_cntx);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B pack time: %f \n", diff_t ); 
	}


	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
	// otherwise just allocate an empty C_p buffer
	if(beta != 0) {
		clock_gettime(CLOCK_REALTIME, &start);
	    C_sz = cake_sgemm_packed_C_size(M, N, 1, x, cake_cntx);
		if(posix_memalign((void**) &C_p, 64, C_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}
	    // C_sz = cake_sgemm_packed_C_size(M, N, 1, cake_cntx) / sizeof(float);
	    // C_p = (float*) calloc(C_sz, sizeof(float));
		pack_C_single_buf_m_first(C, C_p, M, N, p, x, cake_cntx);

		// C_p = (float**) malloc((((M / (p*m_c))*p) + p_l) * (N/n_c + n_pad) * sizeof( float* ));
		// pack_C(C, C_p, M, N, m_c, n_c, m_r, n_r, p, alpha_n);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("C pack time: %f \n", diff_t ); 

	} else {
	    C_sz = cake_sgemm_packed_C_size(M, N, 1, x, cake_cntx);
	    C_p = (float*) calloc(C_sz / sizeof(float), sizeof(float));
	}



	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int k_c1_last_core = x->k_c1_last_core;
	int k_rem = x->k_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;

	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int k_cb, n_c_t, m_c_t, p_used, core;
	inc_t rsc, csc;

	// Set the scalars to use during each GEMM kernel call.
	float alpha_blis = 1.0;
	float beta_blis  = 1.0;
    // rsc = 1; csc = m_r;
    rsc = n_r; csc = 1;
    // rsa = 1; csa = m_r;
    // rsb = n_r; csb = 1;

	//rsc = 1; csc = m;
	//rsa = 1; csa = m;
	//rsb = 1; csb = k;

	// void (*blis_kernel)(dim_t, float*, float*, float*, 
	// 					float*, float*, inc_t, inc_t, 
	// 					auxinfo_t*, cntx_t*);
	// bli_sgemm_ukernel = bli_sgemm_haswell_asm_6x16;


	// m_start = 0;
	// m_end = Mb;
	// m_inc = 1;

	clock_gettime(CLOCK_REALTIME, &start);

	for(n = 0; n < Nb; n++) {

		if(n % 2) {
			k_start = Kb - 1;
			k_end = -1;
			k_inc = -1;
		} else {
			k_start = 0;
			k_end = Kb;
			k_inc = 1;
		}

		n_c_t = n_c;
		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
		}

		for(k = k_start; k != k_end; k += k_inc) {

			if(n % 2) {
				if(k % 2) {
					m_start = 0;
					m_end = Mb;
					m_inc = 1;
				} else {
					m_start = Mb - 1;
					m_end = -1;
					m_inc = -1;
				}
			} else {
				if(k % 2) {
					m_start = Mb - 1;
					m_end = -1;
					m_inc = -1;
				} else {
					m_start = 0;
					m_end = Mb;
					m_inc = 1;
				}
			}

			if((k == Kb - 1) && k_pad) {
				p_used = p_l;
				k_cb = k_rem; 
			} else {
				p_used = p;
				k_cb = p_used*k_c;
			}

			#pragma omp parallel for private(m, m_c_t, core)
			for(int m_ind = 0; m_ind < Mb; m_ind++) {
			// for(m = 0; m < Mb; m++) {

				m = m_start + m_ind*m_inc;

				m_c_t = m_c; 
				if((m == Mb - 1) && m_pad) {
					m_c_t = m_c1;
				}

				// #pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int k_c_t, k_c_x, n_reg, m_reg;

					if((k == Kb - 1) && k_pad) {
						k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
						k_c_x = k_c1;
					} else {
						k_c_t = k_c;
						k_c_x = k_c;
					}

					int a_ind = k*M_padded*p*k_c + m*m_c*k_cb + core*m_c_t*k_c_x;
					int b_ind = n*K*n_c + k*p*k_c*n_c_t + core*k_c_x*n_c_t;
					int c_ind = n*M_padded*n_c + m*m_c*n_c_t;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							
							// bli_sgemm_haswell_asm_6x16(k_c_t, &alpha_blis, 
					  //  		&A_p[a_ind + m_reg*m_r*k_c_t], 
					  //  		&B_p[b_ind + n_reg*k_c_t*n_r], &beta_blis, 
					  //  		&C_p_loc[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
					  //  		rsc, csc, NULL, cake_cntx->blis_cntx);
							bli_sgemm_haswell_asm_6x16(k_c_t, &alpha_blis, 
					   		&A_p[a_ind + m_reg*m_r*k_c_t], 
					   		&B_p[b_ind + n_reg*k_c_t*n_r], &beta_blis, 
					   		&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
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

	// unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 
	unpack_C_single_buf_m_first(C, C_p, M, N, p, x, cake_cntx); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);


	if(!packedA) free(A_p);
	if(!packedB) free(B_p);
	free(C_p);

	return times;
}
