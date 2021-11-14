#include "cake.h"
  


void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {

   // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

   int m_c = x->m_c, n_c = x->n_c;
   int m_c1 = x->m_c1, n_c1 = x->n_c1;
   int m_pad = x->m_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Nb = x->Nb;
   int M_padded = x->M_padded;

   int m, n, n_c_t, n1, C_offset = 0, C_p_offset = 0;

   for(n = 0; n < Nb; n++) {

      if((n == Nb - 1) && n_pad) {
         n_c_t = n_c1;
         n1 = (N - (N % n_c));
      } else {
         n_c_t = n_c;
         n1 = n*n_c;
      }

      C_p_offset = M_padded*n*n_c;

      #pragma omp parallel for private(m, C_offset)
      for(m = 0; m < Mb; m++) {

         int m_c_t, m1;

         if((m == Mb - 1) && m_pad) {
            m_c_t = m_c1;
            m1 = (M - (M % m_c));
         } else {
            m_c_t = m_c;
            m1 = m*m_c;
         }

         C_offset = m*m_c*N + n*n_c;

         unpack_ob_C_single_buf(&C[C_offset], &C_p[C_p_offset + m*m_c*n_c_t], 
            M, N, m1, n1, 0, m_c_t, n_c_t, m_r, n_r);
      }
   }
}

