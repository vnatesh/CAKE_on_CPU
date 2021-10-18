#include "cake.h"
  


void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {

   struct timespec start, end;
   double diff_t;
   clock_gettime(CLOCK_REALTIME, &start);

   int m_c = blk_dims->m_c;
   int n_c = blk_dims->n_c;
   int m_r = cake_cntx->mr;
   int n_r = cake_cntx->nr;

   int m_pad = (M % m_c) ? 1 : 0; 
   int n_pad = (N % n_c) ? 1 : 0;
   int Mb = (M / m_c) + m_pad;
   int Nb = (N / n_c) + n_pad;

   int mr_rem = (int) ceil( ((double) (M % m_c)) / m_r);
   int m_c1 = mr_rem * m_r;

   int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
   int n_c1 = nr_rem * n_r;

   int m, n, n_c_t, n1, C_offset = 0, C_p_offset = 0;
   bool pad_n;

   int M_padded = (M / m_c)*m_c + m_c1;

   for(n = 0; n < Nb; n++) {

      if((n == Nb - 1) && n_pad) {
         n_c_t = n_c1;
         n1 = (N - (N % n_c));
         pad_n = 1;
      } else {
         n_c_t = n_c;
         n1 = n*n_c;
         pad_n = 0;
      }

      C_p_offset = M_padded*n*n_c;

      #pragma omp parallel for private(m, C_offset)
      for(m = 0; m < Mb; m++) {

         int m_c_t, m1;
         bool pad_m;

         if((m == Mb - 1) && m_pad) {
            m_c_t = m_c1;
            m1 = (M - (M % m_c));
            pad_m = 1;
         } else {
            m_c_t = m_c;
            m1 = m*m_c;
            pad_m = 0;
         }

         C_offset = m*m_c*N + n*n_c;

         unpack_ob_C_single_buf(&C[C_offset], &C_p[C_p_offset + m*m_c*n_c_t], 
            M, N, m1, n1, 0, m_c_t, n_c_t, m_r, n_r);
      }
   }
}

