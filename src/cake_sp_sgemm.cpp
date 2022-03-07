#include "cake.h"


void schedule_KMN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;

	int m, k, n; //, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int m_cb, n_c_t, p_used, core;

    // rsc = 1; csc = m_r;
	float* A_p = sp_pack->A_sp_p;
	int* nnz_outer = sp_pack->nnz_outer;
	int* k_inds = sp_pack->k_inds;
	int* loc_m = sp_pack->loc_m;


	for(n = 0; n < Nb; n++) {

		n_c_t = n_c;
		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
		}

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
				int m_c_t, m_c_x, k_c_t, n_reg, m_reg;

				if((m == Mb - 1) && m_pad) {
					m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
					m_c_x = m_c1;
				} else {
					m_c_t = m_c;
					m_c_x = m_c; 
				}

				// pragma omp also here possible (j_r loop)
				// for(k = k_start; k != k_end; k += k_inc) {
				for(k = 0; k < Kb; k++) {
					
					k_c_t = k_c; 
					if((k == Kb - 1) && k_pad) {
						k_c_t = k_c1;
					}

					int out_ind = (m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t) / m_r;
					int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
					// int a_ind = ((m*p*m_c*Kb + k*m_cb) / m_r) + core*mr_per_core;
					int b_ind = n*K*n_c + k*k_c*n_c_t;
					int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;
					
					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {

						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {							

							// A_p_offset = nnz_outer_blk[a_ind + m_reg];
							cake_sp_sgemm_haswell_6x16(&A_p[a_ind + m_reg*m_r*k_c_t], 
													&B_p[b_ind + n_reg*k_c_t*n_r], 
													&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
													m_r, n_r, k_c_t, 
													&nnz_outer[out_ind + m_reg*k_c_t],
													&k_inds[out_ind + m_reg*k_c_t], 
													&loc_m[a_ind + m_reg*m_r*k_c_t]);
						}
					}
					// ob_offset += nnz_ob[m*p*Kb + k*p_used + core];
				}
			}
		}
	}
}

