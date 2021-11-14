#include "cake.h"


// pack the entire matrix A into a single cache-aligned buffer
double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {


      // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int k_c1_last_core = x->k_c1_last_core;
   int k_rem = x->k_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;


   struct timespec start, end;
   double diff_t;
   clock_gettime(CLOCK_REALTIME, &start);

   int m, k, A_offset = 0, A_p_offset = 0;
   int k_cb, m_c_t, p_used, core;


   for(k = 0; k < Kb; k++) {

      if((k == Kb - 1) && k_pad) {
         p_used = p_l;
         k_cb = k_rem; 
      } else {
         p_used = p;
         k_cb = p_used*k_c;
      }

      for(m = 0; m < Mb; m++) {
         
         m_c_t = m_c; 
         if((m == Mb - 1) && m_pad) {
            m_c_t = m_c1;
         }

         A_offset = m*m_c*K + k*p*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int k_c_t, k_c_x;
            bool pad;

            if((k == Kb - 1) && k_pad) {
               k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
               k_c_x = k_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               k_c_t = k_c;
               k_c_x = k_c;
               pad = 0;
            }

            pack_ob_A_single_buf(&A[A_offset + core*k_c_x], &A_p[A_p_offset + core*k_c_x*m_c_t], 
               M, K, m*m_c, 0, m_c_t, k_c_t, m_r, pad);
         }

         A_p_offset += k_cb*m_c_t;
      }
   }

     clock_gettime(CLOCK_REALTIME, &end);
     long seconds = end.tv_sec - start.tv_sec;
     long nanoseconds = end.tv_nsec - start.tv_nsec;
     diff_t = seconds + nanoseconds*1e-9;

     return diff_t;
}




void pack_B_m_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {


      // copy over block dims to local vars to avoid readibility ussiues with x->
   int n_r = cake_cntx->nr;

   int k_c = x->k_c, n_c = x->n_c;
   int k_c1 = x->k_c1, n_c1 = x->n_c1;
   int k_c1_last_core = x->k_c1_last_core;
   int k_rem = x->k_rem;
   int p_l = x->p_l, k_pad = x->k_pad, n_pad = x->n_pad;
   int Kb = x->Kb, Nb = x->Nb;

   int n, n1, k, B_offset = 0, B_p_offset = 0;
   int k_cb, n_c_t, p_used, core;

   for(n = 0; n < Nb; n++) {

      if((n == Nb - 1) && n_pad) {
         n_c_t = n_c1;
         n1 = (N - (N % n_c));
      } else {
         n_c_t = n_c;
         n1 = n*n_c;
      }

      for(k = 0; k < Kb; k++) {

         if((k == Kb - 1) && k_pad) {
            p_used = p_l;
            k_cb = k_rem; 
         } else {
            p_used = p;
            k_cb = p_used*k_c;
         }

         B_offset = k*p*k_c*N + n*n_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int k_c_t, k_c_x;
            bool pad;

            if((k == Kb - 1) && k_pad) {
               k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
               k_c_x = k_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               k_c_t = k_c;
               k_c_x = k_c;
               pad = 0;
            }

            pack_ob_B_single_buf(&B[B_offset + core*k_c_x*N], &B_p[B_p_offset + core*k_c_x*n_c_t], 
               K, N, n1, k_c_t, n_c_t, n_r, pad);
         }

         B_p_offset += k_cb*n_c_t;
      }
   }
}



void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {


      // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

   int m_c = x->m_c, n_c = x->n_c;
   int m_c1 = x->m_c1, n_c1 = x->n_c1;
   int m_pad = x->m_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Nb = x->Nb;
   int M_padded = x->M_padded;

   int m, n, n_c_t, n1, C_offset = 0, C_p_offset = 0;
   bool pad_n;


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

         pack_ob_C_single_buf(&C[C_offset], &C_p[C_p_offset + m*m_c*n_c_t], 
            M, N, m1, n1, 0, m_c_t, n_c_t, m_r, n_r, pad_m, pad_n);
      }
   }
}




