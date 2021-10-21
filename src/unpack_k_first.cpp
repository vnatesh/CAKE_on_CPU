#include "cake.h"
  



void unpack_C_single_buf(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx) {

   struct timespec start, end;
   double diff_t;
   clock_gettime(CLOCK_REALTIME, &start);

   int m_r = cake_cntx->mr;
   int n_r = cake_cntx->nr;

   int m, n, C_offset = 0, C_p_offset = 0;
   int m_cb, n_c_t, p_used, core;

   int m1 = (M - (M % (p*m_c)));
   int n1 = (N - (N % n_c));
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
            bool pad_m;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
               pad_m = (core == (p_l - 1) ? 1 : 0);
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
               pad_m = 0;
            }

            unpack_ob_C_single_buf(&C[C_offset + core*m_c_x*N], &C_p[C_p_offset + core*m_c_x*n_c_t], 
               M, N, m1, n1, core*m_c_x, m_c_t, n_c_t, m_r, n_r);
         }

         C_p_offset += m_cb*n_c_t;
      }
   }
}





void unpack_C_rsc_multiple_buf(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n) {

	int m1, m2, n1;
	int ind1 = 0;


	for(n1 = 0; n1 < (N - (N%n_c)); n1 += n_c) {
		for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {
				unpack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c], M, N, m1, n1, m2, m_c, n_c, m_r, n_r);
			}
			ind1 += p;
		}

		if(M % (p*m_c)) {	

			m1 = (M - (M % (p*m_c)));
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {
				unpack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c1], M, N, m1, n1, m2, m_c1, n_c, m_r, n_r);
			}
			ind1 += (p_l-1);

			m2 = (p_l-1) * m_c1;
			unpack_ob_C_multiple_buf(C, C_p[ind1], M, N, m1, n1, m2, m_c1_last_core, n_c, m_r, n_r);
			ind1++;
		}
	}

	// right-most column of CBS blocks with p OBs of shape m_c x n_c1
	n1 = (N - (N%n_c));

	if(n_c1) {	

		for(m1 = 0; m1 < (M - (M % (p*m_c))); m1 += p*m_c) {
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < p*m_c; m2 += m_c) {
				unpack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c], M, N, m1, n1, m2, m_c, n_c1, m_r, n_r);
			}

			ind1 += p;
		}

		if(M % (p*m_c)) {	

			m1 = (M - (M % (p*m_c)));

			// last row of CBS blocks with p_l-1 m_c1 x n_c1 OBs and 1 m_c1_last_core x n_c1 OB
			#pragma omp parallel for private(m2)
			for(m2 = 0; m2 < (p_l-1)*m_c1; m2 += m_c1) {
				unpack_ob_C_multiple_buf(C, C_p[ind1 + m2/m_c1], M, N, m1, n1, m2, m_c1, n_c1, m_r, n_r);
			}

			ind1 += (p_l-1);

			// last OB in C (lower right corner) with shape m_c1_last_core * n_c1
			m2 = (p_l-1) * m_c1;
			unpack_ob_C_multiple_buf(C, C_p[ind1], M, N, m1, n1, m2, m_c1_last_core, n_c1, m_r, n_r);
			ind1++;
		}
	}
}


