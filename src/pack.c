#include "cake.h"
 

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

