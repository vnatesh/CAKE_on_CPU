#include "cake.h"


// pack the entire matrix A into a single cache-aligned buffer
double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {

	struct timespec start, end;
	double diff_t;
	clock_gettime(CLOCK_REALTIME, &start);

	int m_c = blk_dims->m_c;
	int k_c = blk_dims->k_c;
	int m_r = cake_cntx->mr;

   int k_pad = (K % (p*k_c)) ? 1 : 0; 
   int m_pad = (M % m_c) ? 1 : 0; 
   int Mb = (M / m_c) + m_pad;
   int Kb = (K / (p*k_c)) + k_pad;

   int k_rem = K % (p*k_c);
   int k_c1 = (int) ceil( ((double) k_rem) / p);
   int p_l;

   if(k_c1) 
      p_l = (int) ceil( ((double) k_rem) / k_c1);
   else
      p_l = 0;

   int k_c1_last_core = k_rem - k_c1*(p_l-1);
   int mr_rem = (int) ceil( ((double) (M % m_c)) / m_r);
   int m_c1 = mr_rem * m_r;

   int m, k, A_offset = 0, A_p_offset = 0;
   int k_cb, m_c_t, p_used, core;


   for(k = 0; k < Mb; k++) {

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




void pack_B_m_first(float* B, float* B_p, int K, int N, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {

   int k_c = blk_dims->k_c;
   int n_c = blk_dims->n_c;
   int n_r = cake_cntx->nr;

   int k_pad = (K % (p*k_c)) ? 1 : 0; 
   int n_pad = (N % n_c) ? 1 : 0;
   int Kb = (K / (p*k_c)) + k_pad;
   int Nb = (N / n_c) + n_pad;

   int k_rem = K % (p*k_c);
   int k_c1 = (int) ceil( ((double) k_rem) / p);
   int p_l;

   if(k_c1) 
      p_l = (int) ceil( ((double) k_rem) / k_c1);
   else
      p_l = 0;

   int k_c1_last_core = k_rem - k_c1*(p_l-1);
   int nr_rem = (int) ceil( ((double) (N % n_c) / n_r)) ;
   int n_c1 = nr_rem * n_r;

   int n, n1, k, B_offset = 0, B_p_offset = 0;
   int k_cb, n_c_t, p_used, core;
   bool pad_n;

   int ind1 = 0;

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
               K, N, k_c_t, n_c_t, n_r, pad);
         }

         B_p_offset += k_cb*n_c_t;
      }
   }
}



void pack_ob_B_single_buf(float* B, float* B_p, int K, int N,
            int k_c, int n_c, int n_r, bool pad_n) {

   int ind_ob = 0;

   if(pad_n) {
      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) <  N) {
                  // B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = B[n1 + k1*N + n2 + i*N + j];
                  B_p[ind_ob] = B[n2 + i*N + j];
               }
               ind_ob++;
            }
         }
      }
   }
   else {
      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               B_p[ind_ob] = B[n2 + i*N + j];
               ind_ob++;
            }
         }
      }
   }
}



void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx, blk_dims_t* blk_dims) {

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

   int m, n, n_c_t, n1 C_offset = 0, C_p_offset = 0;
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

         pack_ob_C_single_buf(&C[C_offset], &C_p[C_p_offset + m*m_c*n_c_t], 
            M, N, m1, n1, 0, m_c_t, n_c_t, m_r, n_r, pad_m, pad_n);
      }
   }
}




void pack_ob_C_single_buf_m_first(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n) {

   int ind_ob = 0;

   if(pad_m || pad_n) {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  if((n1 + n2 + j) < N  ||  (m1 + m2 + m3 + i) <  M) {
                     C_p[ind_ob] = C[n2 + m3*N + i*N + j];
                  }
                  ind_ob++;
               }
            }
         }
      }

   } else {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  C_p[ind_ob] = C[n2 + m3*N + i*N + j];
                  ind_ob++;
               }
            }
         }
      }
   }
}



void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int ind_ob = 0;

   if(pad) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < m_r; j++) {
               if((m1 + m2 + m3 + j) <  M) {
                  A_p[ind_ob] = A[m3*K + i + j*K];
               }

               ind_ob++;
            }
         }
      }     
   } 

   else {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < m_r; j++) {
               A_p[ind_ob] = A[m3*K + i + j*K];
               ind_ob++;
            }
         }
      }     
   }
}