#include "cake.h"


void schedule_KMN(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;

	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m_cb, n_c_t, p_used, core;

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
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

							cake_sgemm_ukernel(&A_p[a_ind + m_reg*m_r*k_c_t], 
											&B_p[b_ind + n_reg*k_c_t*n_r], 
											&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
											m_r, n_r, k_c_t, cake_cntx);

							// kernel_map[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
							// 				&B_p[b_ind + n_reg*k_c_t*n_r], 
							// 				&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
							// 				m_r, n_r, k_c_t);
						}
					}
				}
			}
		}
	}
}








void schedule_KMN_2d(float* A, float* B, float* C, float* A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c, pm = x->pm, pn = x->pn;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Kb = x->Kb;
 
	int k, n2, m3, k_c_t, core, p_used;
	float kappa = 1.0;
	int lda = 1;
	p_used = p;


	int M_padded = x->M_padded;
	int N_padded = x->N_padded;


	for(k = 0; k < Kb; k++) {
		
		k_c_t = k_c; 
		if((k == Kb - 1) && k_pad) {
			k_c_t = k_c1;
		}

	// if(n_pad) {
		// pack_ob_B_parallel(&B[k*k_c*N], B_p, K, N, 0, k_c_t, N_padded, n_r, n_pad);
	// } else {
    	#pragma omp parallel for private(n2)
      	for(n2 = 0; n2 < N_padded; n2 += n_r) {
         	bli_spackm_haswell_asm_16xk(n_r, k_c_t, &kappa, &B[k*k_c*N + n2], lda, N, 
                                    &B_p[n2*k_c_t], n_r);
      	}
   	// }

    // if(m_pad) {
        // pack_ob_A_parallel(&A[k*k_c], A_p, M, K, 0, 0, M_padded, k_c_t, m_r, m_pad);
   	// } else {          
    	#pragma omp parallel for private(m3)
		for(m3 = 0; m3 < M_padded; m3 += m_r) {
			bli_spackm_haswell_asm_6xk(m_r, k_c_t, &kappa, &A[k*k_c + m3*K], K, lda, 
				&A_p[m3*k_c_t], m_r);
		}     
   	// }

		// pragma omp here (i_c loop)
		#pragma omp parallel for private(core)
		for(core = 0; core < p_used; core++) {

			int n1, n2, n_c_t;
			bool pad_n;

			if(((core % pn) == (pn - 1)) && n_pad) {
				n_c_t = n_c1;
				n1 = (N - (N % n_c));
				pad_n = 1;
			} else {
				n_c_t = n_c;
				n1 = (core % pn)*n_c;
				pad_n = 0;
			}

			// These vars must be private to thread, 
			// otherwise out of bounds memory access possible
			int m_c_t, n_reg, m_reg;
			bool pad; 

			if(((core / pn) == (pm - 1)) && m_pad) {
				m_c_t = m_c1;
				pad = 1;
			} else {
				m_c_t = m_c;
				pad = 0;
			}

			for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
				for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

					kernel_map[m_map][n_map](
						&A_p[(core / pn)*m_c*k_c_t + m_reg*m_r*k_c_t], 
						&B_p[(core % pn)*n_c*k_c_t + n_reg*k_c_t*n_r], 
						&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
						m_r, n_r, k_c_t
					);
       // printf("core = %d, n_c_t = %d, m_c_t = %d\n",core, n_c_t, m_c_t );

					// cake_sgemm_ukernel(&A_p[core][m_reg*m_r*k_c_t], 
					// 				&B_p[n_reg*k_c_t*n_r], 
					// 				&C_p[core][n_reg*n_r + m_reg*m_r*n_r], 
					// 				m_r, n_r, k_c_t, cake_cntx);
				}
			}
		}
	}

	// unpack_C
	#pragma omp parallel for private(core)
	for(core = 0; core < p_used; core++) {

		int m_c_t = (((core / pn) == (pm - 1)) && m_pad) ? m_c1 : m_c;
		int n_c_t = (((core % pn) == (pn - 1)) && n_pad) ? n_c1 : n_c; 

		unpack_ob_C_single_buf(&C[(core / pn)*m_c*N + (core % pn)*n_c], C_p[core], 
			M, N, 0, (core % pn)*n_c, (core / pn)*m_c, m_c_t, n_c_t, m_r, n_r);

		memset(C_p[core], 0, m_c * n_c * sizeof(float));
	}
}






// void schedule_KMN_2d(float* A, float* B, float* C, float** A_p, float** B_p, float** C_p, int M, int N, int K, int p, 
// 	cake_cntx_t* cake_cntx, blk_dims_t* x) {

// 	// copy over block dims to local vars to avoid readibility ussiues with x->
// 	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
// 	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

// 	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c, pm = x->pm, pn = x->pn;
// 	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
// 	int m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
// 	int Kb = x->Kb;
 
// 	int k, k_c_t, core, p_used;
// 	float kappa = 1.0;
// 	int lda = 1;
// 	p_used = p;


// 	for(k = 0; k < Kb; k++) {
		
// 		k_c_t = k_c; 
// 		if((k == Kb - 1) && k_pad) {
// 			k_c_t = k_c1;
// 		}

// 		// pragma omp here (i_c loop)
// 		#pragma omp parallel for private(core)
// 		for(core = 0; core < p_used; core++) {

// 			int n1, n2, n_c_t;
// 			bool pad_n;

// 			if(((core % pn) == (pn - 1)) && n_pad) {
// 				n_c_t = n_c1;
// 				n1 = (N - (N % n_c));
// 				pad_n = 1;
// 			} else {
// 				n_c_t = n_c;
// 				n1 = (core % pn)*n_c;
// 				pad_n = 0;
// 			}

// 			// pack_ob_B_parallel(&B[n1 + k*k_c*N], B_p, K, N, n1, k_c_t, n_c_t, n_r, pad_n);
// 			if(pad_n) {
// 				// pack_ob_B_single_buf(&B[n1 + k*k_c*N], &B_p[(core % pn)*n_c*k_c_t], K, N, n1, k_c_t, n_c_t, n_r, pad_n);
// 				pack_ob_B_single_buf(&B[n1 + k*k_c*N], B_p[core], K, N, n1, k_c_t, n_c_t, n_r, pad_n);
// 			} else {
// 		      	for(n2 = 0; n2 < n_c; n2 += n_r) {
// 		        	 // bli_spackm_haswell_asm_16xk(n_r, k_c, &kappa, &B[n1 + k1*N + n2], lda, N, 
// 		         	//                            &B_p[ind1 + (k1/k_c)*k_c*n_c + n2*k_c], n_r);
// 		         	bli_spackm_haswell_asm_16xk(n_r, k_c_t, &kappa, &B[n1 + k*k_c*N + n2], lda, N, 
// 		                                    &B_p[core][n2*k_c_t], n_r);
// 		      	}
// 		   	}

// 			// These vars must be private to thread, 
// 			// otherwise out of bounds memory access possible
// 			int m_c_t, n_reg, m_reg;
// 			bool pad; 

// 			if(((core / pn) == (pm - 1)) && m_pad) {
// 				m_c_t = m_c1;
// 				pad = 1;
// 			} else {
// 				m_c_t = m_c;
// 				pad = 0;
// 			}

// 	            // pack_ob_A_single_buf(&A[A_offset + core*m_c_x*K], A_p[core], 
// 	            //    M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
//             if(pad) {
// 	            pack_ob_A_single_buf(&A[k*k_c + (core / pn)*m_c*K], A_p[core], 
// 	               M, K, 0, (core / pn)*m_c, m_c_t, k_c_t, m_r, pad);
//            	} else {          
// 				for(int m3 = 0; m3 < m_c_t; m3 += m_r) {
// 					bli_spackm_haswell_asm_6xk(m_r, k_c_t, &kappa, &A[k*k_c + (core / pn)*m_c*K + m3*K], K, lda, 
// 						&A_p[core][m3*k_c_t], m_r);
// 				}     
//            	}

// 			for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
// 				for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

// 					kernel_map[m_map][n_map](
// 						&A_p[core][m_reg*m_r*k_c_t], 
// 						&B_p[core][n_reg*k_c_t*n_r], 
// 						&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
// 						m_r, n_r, k_c_t
// 					);
//        printf("core = %d, n_c_t = %d, m_c_t = %d\n",core, n_c_t, m_c_t );

// 					// cake_sgemm_ukernel(&A_p[core][m_reg*m_r*k_c_t], 
// 					// 				&B_p[n_reg*k_c_t*n_r], 
// 					// 				&C_p[core][n_reg*n_r + m_reg*m_r*n_r], 
// 					// 				m_r, n_r, k_c_t, cake_cntx);
// 				}
// 			}
// 		}
// 	}

// 	// unpack_C
// 	#pragma omp parallel for private(core)
// 	for(core = 0; core < p_used; core++) {

// 		int m_c_t = (((core / pn) == (pm - 1)) && m_pad) ? m_c1 : m_c;
// 		int n_c_t = (((core % pn) == (pn - 1)) && n_pad) ? n_c1 : n_c; 
// 				// n1 = (N - (N % n_c));

// 				// n1 = (core % pn)*n_c;


// 		unpack_ob_C_single_buf(&C[(core / pn)*m_c*N + (core % pn)*n_c], C_p[core], 
// 			M, N, 0, (core % pn)*n_c, (core / pn)*m_c, m_c_t, n_c_t, m_r, n_r);

// 		memset(C_p[core], 0, m_c * n_c * sizeof(float));
// 	}
// }







void schedule_KMN_online(float* A, float* B, float* C, float** A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;
 
	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m_cb, n_c_t, p_used, core;

	int m1, n1, n2, A_offset = 0, C_offset = 0, C_p_offset = 0;
	bool pad_n;

	float kappa = 1.0;
	int lda = 1;



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


		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
			n1 = (N - (N % n_c));
			pad_n = 1;
		} else {
			n_c_t = n_c;
			n1 = n*n_c;
			pad_n = 0;
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
				m1 = (M - (M % (p*m_c)));
			} else {
				p_used = p;
				m_cb = p_used*m_c;
				m1 = m*p*m_c;
			}

			// pragma omp also here possible (j_r loop)
			for(k = k_start; k != k_end; k += k_inc) {
				
				int k_c_t;

				k_c_t = k_c; 
				if((k == Kb - 1) && k_pad) {
					k_c_t = k_c1;
				}


				if((k == k_start) && (m != m_start))  {
				 	// skip packing at the corner turns
					// printf("reuse B\n");
				} else {

					// pack_ob_B_parallel(&B[n1 + k*k_c*N], B_p, K, N, n1, k_c_t, n_c_t, n_r, pad_n);
					if(pad_n) {
						pack_ob_B_parallel(&B[n1 + k*k_c*N], B_p, K, N, n1, k_c_t, n_c_t, n_r, pad_n);
					} else {
				    	#pragma omp parallel for private(n2)
				      	for(n2 = 0; n2 < n_c; n2 += n_r) {
				        	 // bli_spackm_haswell_asm_16xk(n_r, k_c, &kappa, &B[n1 + k1*N + n2], lda, N, 
				         	//                            &B_p[ind1 + (k1/k_c)*k_c*n_c + n2*k_c], n_r);
				         	bli_spackm_haswell_asm_16xk(n_r, k_c_t, &kappa, &B[n1 + k*k_c*N + n2], lda, N, 
				                                    &B_p[n2*k_c_t], n_r);
				      	}
				   	}
				}


				// pragma omp here (i_c loop)
				#pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int m_c_t, m_c_x, n_reg, m_reg;
					bool pad; 

					if((m == Mb - 1) && m_pad) {
						m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
						m_c_x = m_c1;
						pad = (core == (p_l - 1) ? 1 : 0);
					} else {
						m_c_t = m_c;
						m_c_x = m_c; 
						pad = 0;
					}

					A_offset = m*p*m_c*K + k*k_c;

			            // pack_ob_A_single_buf(&A[A_offset + core*m_c_x*K], A_p[core], 
			            //    M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
					if((k == k_start) && (m == m_start) && (n != 0))  {
						// printf("reuse A\n");
					} else {
			            if(pad) {
				            pack_ob_A_single_buf(&A[A_offset + core*m_c_x*K], A_p[core], 
				               M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
			           	} else {          
							for(int m3 = 0; m3 < m_c_t; m3 += m_r) {
								bli_spackm_haswell_asm_6xk(m_r, k_c_t, &kappa, &A[A_offset + core*m_c_x*K + m3*K], K, lda, 
									&A_p[core][m3*k_c_t], m_r);
							}     
			           	}
		            }


					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

							kernel_map[m_map][n_map](
								&A_p[core][m_reg*m_r*k_c_t], 
								&B_p[n_reg*k_c_t*n_r], 
								&C_p[core][n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
								m_r, n_r, k_c_t
							);

							// cake_sgemm_ukernel(&A_p[core][m_reg*m_r*k_c_t], 
							// 				&B_p[n_reg*k_c_t*n_r], 
							// 				&C_p[core][n_reg*n_r + m_reg*m_r*n_r], 
							// 				m_r, n_r, k_c_t, cake_cntx);
						}
					}
				}
			}

			// unpack_C

			C_offset = m*p*m_c*N + n*n_c;

			#pragma omp parallel for private(core)
			for(core = 0; core < p_used; core++) {

				int m_c_t, m_c_x;

				if((m == Mb - 1) && m_pad) {
					m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
					m_c_x = m_c1;
				} else {
					m_c_t = m_c;
					m_c_x = m_c;
				}

				unpack_ob_C_single_buf(&C[C_offset + core*m_c_x*N], C_p[core], 
					M, N, m1, n1, core*m_c_x, m_c_t, n_c_t, m_r, n_r);

				memset(C_p[core], 0, m_c * n_c * sizeof(float));
			}
		}
	}
}



void schedule_KMN_C_unpacked(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;
	int N_padded = x->N_padded;

	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m_cb, n_c_t, p_used, core;

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
					// int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;
					int c_ind = n*n_c + m*p*m_c*N_padded + core*m_c_x*N_padded;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

							// kernel_map[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
							// 				&B_p[b_ind + n_reg*k_c_t*n_r], 
							// 				&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
							// 				m_r, n_r, k_c_t);
							cake_sgemm_ukernel(&A_p[a_ind + m_reg*m_r*k_c_t], 
											&B_p[b_ind + n_reg*k_c_t*n_r], 
											&C_p[c_ind + n_reg*n_r + m_reg*m_r*N_padded], 
											m_r, N_padded, k_c_t, cake_cntx);


						}
					}
				}
			}
		}
	}
}


