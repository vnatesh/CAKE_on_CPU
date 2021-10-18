#include "cake.h"



int cake_sgemm_packed_A_size(int M, int K, int p, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {

	int mr_rem = (int) ceil( ((double) (M % (p*blk_dims->m_c))) / cake_cntx->mr) ;
	int M_padded = (cake_cntx->mr*mr_rem + (M /(p*blk_dims->m_c))*p*blk_dims->m_c);

	return (M_padded * K) * sizeof(float);
}



int cake_sgemm_packed_B_size(int K, int N, int p, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {
	
	int nr_rem = (int) ceil( ((double) (N % blk_dims->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N%blk_dims->n_c)) + n_c1;

	return (K * N_padded) * sizeof(float);
}



int cake_sgemm_packed_C_size(int M, int N, int p, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {

	int mr_rem = (int) ceil( ((double) (M % (p*blk_dims->m_c))) / cake_cntx->mr) ;
	int M_padded = (cake_cntx->mr*mr_rem + (M /(p*blk_dims->m_c))*p*blk_dims->m_c);

	int nr_rem = (int) ceil( ((double) (N % blk_dims->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N%blk_dims->n_c)) + n_c1;

	return (M_padded * N_padded) * sizeof(float);
}




// pack each operation block (OB) of matrix A into its own cache-aligned buffer
double pack_A_multiple_buf(float* A, float** A_p, int M, int K, int m_c, int k_c, int m_r, int p) {
	struct timespec start, end;
	double diff_t;
	clock_gettime(CLOCK_REALTIME, &start);

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

			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {
				// printf("hey %d\n", ind1 + m2/m_c);
				if(posix_memalign((void**) &A_p[ind1 + m2/m_c], 64, m_c * k_c * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_A_multiple_buf(A, A_p[ind1 + m2/m_c], M, K, m1, k1, m2, m_c, k_c, m_r, 0);

				if(ARR_PRINT) print_array(A_p[ind1 + m2/m_c], k_c * m_c);
			}

			ind1 += p;
		}
		// right-most column of CBS blocks each with p m_c x k_c1 OBs
		if(k_c1) {
			k1 = K - (K%k_c);
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				if(posix_memalign((void**) &A_p[ind1 + m2/m_c], 64, k_c1 * m_c * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_A_multiple_buf(A, A_p[ind1 + m2/m_c], M, K, m1, k1, m2, m_c, k_c1, m_r, 0);
				if(ARR_PRINT) print_array(A_p[ind1 + m2/m_c], k_c1 * m_c);
			}
			ind1 += p;
		}
	}

	// Process bottom-most rows of CBS blocks and perform M-dim padding
	if(M % (p*m_c)) {	

		m1 = (M - (M % (p*m_c)));

		for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {
				
				if(posix_memalign((void**) &A_p[ind1 + m2/m_c1], 64, k_c * m_c1 * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_A_multiple_buf(A, A_p[ind1 + m2/m_c1], M, K, m1, k1, m2, m_c1, k_c, m_r, 0);
				if(ARR_PRINT) print_array(A_p[ind1 + m2/m_c1], k_c * m_c1);
				// ind1++;
			}
			ind1 += (p_l-1);

			// final row of CBS blocks each with m_c1_last_core x k_c
			m2 = (p_l-1) * m_c1;
			if(posix_memalign((void**) &A_p[ind1], 64, k_c * m_c1_last_core * sizeof(float))) {
				printf("posix memalign error\n");
				exit(1);
			}

			pack_ob_A_multiple_buf(A, A_p[ind1], M, K, m1, k1, m2, m_c1_last_core, k_c, m_r, 1);
			if(ARR_PRINT) print_array(A_p[ind1], k_c * m_c1_last_core);
			ind1++;
		}

		// Final CBS block (with p_l-1 m_c1 x k_c1 OBs and 1 m_c1_last_core x k_c1 OB) 
		// present in the lower right hand corner of A 
		if(k_c1) {

			k1 = K - (K%k_c);
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {

				if(posix_memalign((void**) &A_p[ind1 + m2/m_c1], 64, k_c1 * m_c1 * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_A_multiple_buf(A, A_p[ind1 + m2/m_c1], M, K, m1, k1, m2, m_c1, k_c1, m_r, 0);
				if(ARR_PRINT) print_array(A_p[ind1 + m2/m_c1], k_c1 * m_c1);
				// ind1++;
			}
			ind1 += (p_l-1);

			// last OB of A has shape m_c1_last_core x k_c1 
			m2 = (p_l-1) * m_c1;
			
			if(posix_memalign((void**) &A_p[ind1], 64, k_c1 * m_c1_last_core * sizeof(float))) {
				printf("posix memalign error\n");
				exit(1);
			}

			pack_ob_A_multiple_buf(A, A_p[ind1], M, K, m1, k1, m2, m_c1_last_core, k_c1, m_r, 1);
			if(ARR_PRINT) print_array(A_p[ind1], k_c1 * m_c1_last_core);
			ind1++;
		}
	}

	clock_gettime(CLOCK_REALTIME, &end);
	long seconds = end.tv_sec - start.tv_sec;
	long nanoseconds = end.tv_nsec - start.tv_nsec;
	diff_t = seconds + nanoseconds*1e-9;

     return diff_t;

}



void pack_C_multiple_buf(float* C, float** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n) {

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
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				if(posix_memalign((void**) &C_p[ind1 + m2/m_c], 64, m_c * n_c * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c], M, N, m1, n1, m2, m_c, n_c, m_r, n_r, 0);
				if(ARR_PRINT) print_array(C_p[ind1 + m2/m_c], m_c * n_c);
				// ind1++;
			}
			ind1 += p;
		}

		// bottom row of CBS blocks with p_l-1 OBs of m_c1 x n_c and 1 OBs of shape m_c1_last_core x n_c
		if(M % (p*m_c)) {	

			m1 = (M - (M % (p*m_c)));
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {

				if(posix_memalign((void**) &C_p[ind1 + m2/m_c1], 64, m_c1 * n_c * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c1], M, N, m1, n1, m2, m_c1, n_c, m_r, n_r, 0);
				if(ARR_PRINT) print_array(C_p[ind1 + m2/m_c1], m_c1 * n_c);
				// ind1++;
			}
			ind1 += (p_l-1);

			m2 = (p_l-1) * m_c1;
			if(posix_memalign((void**) &C_p[ind1], 64, m_c1_last_core * n_c * sizeof(float))) {
				printf("posix memalign error\n");
				exit(1);
			}

			pack_ob_C_multiple_buf(C, C_p[ind1], M, N, m1, n1, m2, m_c1_last_core, n_c, m_r, n_r, 1);
			if(ARR_PRINT) print_array(C_p[ind1], m_c1 * n_c);
			ind1++;
		}
	}

	// right-most column of CBS blocks with p OBs of shape m_c x n_c1
	n1 = (N - (N%n_c));

	if(n_c1) {	

		for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {

				if(posix_memalign((void**) &C_p[ind1 + m2/m_c], 64, m_c * n_c1 * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c], M, N, m1, n1, m2, m_c, n_c1, m_r, n_r, 1);
				if(ARR_PRINT) print_array(C_p[ind1 + m2/m_c], m_c * n_c1);
				// ind1++;
			}
			ind1 += p;
		}

		if(M % (p*m_c)) {	

			m1 = (M - (M % (p*m_c)));

			// last row of CBS blocks with p_l-1 m_c1 x n_c1 OBs and 1 m_c1_last_core x n_c1 OB
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {

				if(posix_memalign((void**) &C_p[ind1 + m2/m_c1], 64, m_c1 * n_c1 * sizeof(float))) {
					printf("posix memalign error\n");
					exit(1);
				}

				pack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c1], M, N, m1, n1, m2, m_c1, n_c1, m_r, n_r, 1);
				if(ARR_PRINT) print_array(C_p[ind1 + m2/m_c1], m_c1 * n_c1);
				// ind1++;
			}
			ind1 += (p_l-1);

			// last OB in C (lower right corner) with shape m_c1_last_core * n_c1
			m2 = (p_l-1) * m_c1;
			if(posix_memalign((void**) &C_p[ind1], 64, m_c1_last_core * n_c1 * sizeof(float))) {
				printf("posix memalign error\n");
				exit(1);
			}

			pack_ob_C_multiple_buf(C, C_p[ind1], M, N, m1, n1, m2, m_c1_last_core, n_c1, m_r, n_r, 1);
			if(ARR_PRINT) print_array(C_p[ind1], m_c1_last_core * n_c1);
			ind1++;
		}
	}
}
