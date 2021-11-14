#include "cake.h"
  


void unpack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {

   // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

   int m_c = x->m_c, n_c = x->n_c;
   int m_c1 = x->m_c1, n_c1 = x->n_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Nb = x->Nb;

   int m, n, C_offset = 0, C_p_offset = 0;
   int m_cb, n_c_t, p_used, core;

   int m1 = (M - (M % (p*m_c)));
   int n1 = (N - (N % n_c));

   for(n = 0; n < Nb; n++) {

      if((n == Nb - 1) && n_pad) {
         n_c_t = n_c1;
         n1 = (N - (N % n_c));
      } else {
         n_c_t = n_c;
         n1 = n*n_c;
      }

      for(m = 0; m < Mb; m++) {

         if((m == Mb - 1) && m_pad) {
            p_used = p_l;
            m_cb = m_r*mr_rem ; 
			m1 = (M - (M % (p*m_c)));
         } else {
            p_used = p;
            m_cb = p_used*m_c;
			m1 = m*p*m_c;
         }

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

            unpack_ob_C_single_buf(&C[C_offset + core*m_c_x*N], &C_p[C_p_offset + core*m_c_x*n_c_t], 
               M, N, m1, n1, core*m_c_x, m_c_t, n_c_t, m_r, n_r);
         }

         C_p_offset += m_cb*n_c_t;
      }
   }
}


