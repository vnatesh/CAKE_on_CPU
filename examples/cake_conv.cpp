#include "cake.h"
#include <immintrin.h>


double cake_sconv(float* A, float* B, float* C, int Wf, int Hf, 
	int Win, int Hin, int Wout, int Hout, int Cin, int Cout, int s, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, 
	enum sched sch = NA);

cache_dims_t* get_cache_dims_conv(int Wf, int Hf, int Win, int Hin, 
	int Wout, int Hout, int Cin, int Cout, int s, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density = 0, float type_size = 4);

void init_block_dims_conv(int Wf, int Hf, int Win, int Hin, 
	int Wout, int Hout, int Cin, int Cout, int s, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density = 0, float type_size = 4);


void print_filters(float* F, int Wf, int Hf, int Cin, int Cout);
void print_feature_map(float* In, int Win, int Hin, int Cin);

float* im_2_col(float* In, int Wf, int Hf, int Win, int Hin, 
	int Wout, int Hout, int Cin, int Cout, int s);

void conv_gemm_checker(float* conv, float* gemm, int M, int N);


// assumes A, B, and C are convolution filters, input, and output, respectively
void cake_conv_haswell_6x16(float* A, float* B, float* C, int n_in, int n_out, int k) {
			  
	__m256 a, b1, b2;
	__m256 c[6*2];


	c[0]  = _mm256_loadu_ps(C);
	c[1]  = _mm256_loadu_ps(C + 8);
	c[2]  = _mm256_loadu_ps(C + n_out);
	c[3]  = _mm256_loadu_ps(C + n_out + 8);
	c[4]  = _mm256_loadu_ps(C + 2*n_out);
	c[5]  = _mm256_loadu_ps(C + 2*n_out + 8);
	c[6]  = _mm256_loadu_ps(C + 3*n_out);
	c[7]  = _mm256_loadu_ps(C + 3*n_out + 8);
	c[8]  = _mm256_loadu_ps(C + 4*n_out);
	c[9]  = _mm256_loadu_ps(C + 4*n_out + 8);
	c[10]  = _mm256_loadu_ps(C + 5*n_out);
	c[11]  = _mm256_loadu_ps(C + 5*n_out + 8);

	// c[0]  = _mm256_loadu_ps(C);
	// c[1]  = _mm256_loadu_ps(C + 8);
	// c[2]  = _mm256_loadu_ps(C + 16);
	// c[3]  = _mm256_loadu_ps(C + 24);
	// c[4]  = _mm256_loadu_ps(C + 32);
	// c[5]  = _mm256_loadu_ps(C + 40);
	// c[6]  = _mm256_loadu_ps(C + 48);
	// c[7]  = _mm256_loadu_ps(C + 56);
	// c[8]  = _mm256_loadu_ps(C + 64);
	// c[9]  = _mm256_loadu_ps(C + 72);
	// c[10]  = _mm256_loadu_ps(C + 80);
	// c[11]  = _mm256_loadu_ps(C + 88);

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

// printf("%f\n", c[11][4]);
		 // _mm_prefetch(B + n_in, 0);
		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

		a = _mm256_broadcast_ss(A++);
		c[0] =  _mm256_fmadd_ps(a, b1, c[0]);
		c[1] =  _mm256_fmadd_ps(a, b2, c[1]);

		a = _mm256_broadcast_ss(A++);
		c[2] =  _mm256_fmadd_ps(a, b1, c[2]);
		c[3] =  _mm256_fmadd_ps(a, b2, c[3]);

		a = _mm256_broadcast_ss(A++);
		c[4] =  _mm256_fmadd_ps(a, b1, c[4]);
		c[5] =  _mm256_fmadd_ps(a, b2, c[5]);

		a = _mm256_broadcast_ss(A++);
		c[6] =  _mm256_fmadd_ps(a, b1, c[6]);
		c[7] =  _mm256_fmadd_ps(a, b2, c[7]);

		a = _mm256_broadcast_ss(A++);
		c[8] =  _mm256_fmadd_ps(a, b1, c[8]);
		c[9] =  _mm256_fmadd_ps(a, b2, c[9]);

		a = _mm256_broadcast_ss(A++);
		c[10] =  _mm256_fmadd_ps(a, b1, c[10]);
		c[11] =  _mm256_fmadd_ps(a, b2, c[11]);

		B += n_in;
		// B += 16;



		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

		a = _mm256_broadcast_ss(A++);
		c[0] =  _mm256_fmadd_ps(a, b1, c[0]);
		c[1] =  _mm256_fmadd_ps(a, b2, c[1]);

		a = _mm256_broadcast_ss(A++);
		c[2] =  _mm256_fmadd_ps(a, b1, c[2]);
		c[3] =  _mm256_fmadd_ps(a, b2, c[3]);

		a = _mm256_broadcast_ss(A++);
		c[4] =  _mm256_fmadd_ps(a, b1, c[4]);
		c[5] =  _mm256_fmadd_ps(a, b2, c[5]);

		a = _mm256_broadcast_ss(A++);
		c[6] =  _mm256_fmadd_ps(a, b1, c[6]);
		c[7] =  _mm256_fmadd_ps(a, b2, c[7]);

		a = _mm256_broadcast_ss(A++);
		c[8] =  _mm256_fmadd_ps(a, b1, c[8]);
		c[9] =  _mm256_fmadd_ps(a, b2, c[9]);

		a = _mm256_broadcast_ss(A++);
		c[10] =  _mm256_fmadd_ps(a, b1, c[10]);
		c[11] =  _mm256_fmadd_ps(a, b2, c[11]);

		B += n_in;
						// B += 16;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

		a = _mm256_broadcast_ss(A++);
		c[0] =  _mm256_fmadd_ps(a, b1, c[0]);
		c[1] =  _mm256_fmadd_ps(a, b2, c[1]);

		a = _mm256_broadcast_ss(A++);
		c[2] =  _mm256_fmadd_ps(a, b1, c[2]);
		c[3] =  _mm256_fmadd_ps(a, b2, c[3]);

		a = _mm256_broadcast_ss(A++);
		c[4] =  _mm256_fmadd_ps(a, b1, c[4]);
		c[5] =  _mm256_fmadd_ps(a, b2, c[5]);

		a = _mm256_broadcast_ss(A++);
		c[6] =  _mm256_fmadd_ps(a, b1, c[6]);
		c[7] =  _mm256_fmadd_ps(a, b2, c[7]);

		a = _mm256_broadcast_ss(A++);
		c[8] =  _mm256_fmadd_ps(a, b1, c[8]);
		c[9] =  _mm256_fmadd_ps(a, b2, c[9]);

		a = _mm256_broadcast_ss(A++);
		c[10] =  _mm256_fmadd_ps(a, b1, c[10]);
		c[11] =  _mm256_fmadd_ps(a, b2, c[11]);

		B += n_in;
						// B += 16;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

		a = _mm256_broadcast_ss(A++);
		c[0] =  _mm256_fmadd_ps(a, b1, c[0]);
		c[1] =  _mm256_fmadd_ps(a, b2, c[1]);

		a = _mm256_broadcast_ss(A++);
		c[2] =  _mm256_fmadd_ps(a, b1, c[2]);
		c[3] =  _mm256_fmadd_ps(a, b2, c[3]);

		a = _mm256_broadcast_ss(A++);
		c[4] =  _mm256_fmadd_ps(a, b1, c[4]);
		c[5] =  _mm256_fmadd_ps(a, b2, c[5]);

		a = _mm256_broadcast_ss(A++);
		c[6] =  _mm256_fmadd_ps(a, b1, c[6]);
		c[7] =  _mm256_fmadd_ps(a, b2, c[7]);

		a = _mm256_broadcast_ss(A++);
		c[8] =  _mm256_fmadd_ps(a, b1, c[8]);
		c[9] =  _mm256_fmadd_ps(a, b2, c[9]);

		a = _mm256_broadcast_ss(A++);
		c[10] =  _mm256_fmadd_ps(a, b1, c[10]);
		c[11] =  _mm256_fmadd_ps(a, b2, c[11]);

		B += n_in;
						// B += 16;


	}

	for(int kk = 0; kk < rem; kk++) { 
			
		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

		a = _mm256_broadcast_ss(A++);
		c[0] =  _mm256_fmadd_ps(a, b1, c[0]);
		c[1] =  _mm256_fmadd_ps(a, b2, c[1]);

		a = _mm256_broadcast_ss(A++);
		c[2] =  _mm256_fmadd_ps(a, b1, c[2]);
		c[3] =  _mm256_fmadd_ps(a, b2, c[3]);

		a = _mm256_broadcast_ss(A++);
		c[4] =  _mm256_fmadd_ps(a, b1, c[4]);
		c[5] =  _mm256_fmadd_ps(a, b2, c[5]);

		a = _mm256_broadcast_ss(A++);
		c[6] =  _mm256_fmadd_ps(a, b1, c[6]);
		c[7] =  _mm256_fmadd_ps(a, b2, c[7]);

		a = _mm256_broadcast_ss(A++);
		c[8] =  _mm256_fmadd_ps(a, b1, c[8]);
		c[9] =  _mm256_fmadd_ps(a, b2, c[9]);

		a = _mm256_broadcast_ss(A++);
		c[10] =  _mm256_fmadd_ps(a, b1, c[10]);
		c[11] =  _mm256_fmadd_ps(a, b2, c[11]);

		B += n_in;
						// B += 16;
	}


	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps((C + 8), c[1]);
	_mm256_storeu_ps((C + n_out), c[2]);
	_mm256_storeu_ps((C + n_out + 8), c[3]);
	_mm256_storeu_ps((C + 2*n_out), c[4]);
	_mm256_storeu_ps((C + 2*n_out + 8), c[5]);
	_mm256_storeu_ps((C + 3*n_out), c[6]);
	_mm256_storeu_ps((C + 3*n_out + 8), c[7]);
	_mm256_storeu_ps((C + 4*n_out), c[8] );
	_mm256_storeu_ps((C + 4*n_out + 8), c[9] );
	_mm256_storeu_ps((C + 5*n_out), c[10]);
	_mm256_storeu_ps((C + 5*n_out + 8), c[11]);

	// _mm256_storeu_ps(C, c[0]);
	// _mm256_storeu_ps((C + 8), c[1]);
	// _mm256_storeu_ps((C + 16), c[2]);
	// _mm256_storeu_ps((C + 24), c[3]);
	// _mm256_storeu_ps((C + 32), c[4]);
	// _mm256_storeu_ps((C + 40), c[5]);
	// _mm256_storeu_ps((C + 48), c[6]);
	// _mm256_storeu_ps((C + 56), c[7]);
	// _mm256_storeu_ps((C + 64), c[8]);
	// _mm256_storeu_ps((C + 72), c[9]);
	// _mm256_storeu_ps((C + 80), c[10]);
	// _mm256_storeu_ps((C + 88), c[11]);

}



cache_dims_t* get_cache_dims_conv(int Wf, int Hf, int Win, int Hin, 
	int Wout, int Hout, int Cin, int Cout, int s, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size) {

	int Cm, Hc, Ck, mc_ret, nc_ret, a, M = Cout;
	int max_threads = cake_cntx->ncores; // 2-way hyperthreaded

	int f_size = Cout*Cin*Wf*Hf;
	int in_size = Cin*Win*Hin;
	int out_size = Cout*Wout*Hout;

    cache_dims_t* blk_ret = (cache_dims_t*) malloc(sizeof(cache_dims_t));

	// if((in_size+f_size+out_size)*4 < cake_cntx->L3) {

	// printf("block sizing\n");
	Cm = (int) ceil( ((double) Cout) /  p );
	Cm -= (Cm % cake_cntx->mr);
	mc_ret = Cm;

	if(M < p*cake_cntx->mr) {
		mc_ret = cake_cntx->mr;
	} else if(M < p*Cm) {
		
		a = (M / p);
		if(a < cake_cntx->mr) {
			mc_ret = cake_cntx->mr;
		} else {
			a += (cake_cntx->mr - (a % cake_cntx->mr));
			mc_ret = a;
		}
	}


	Hc = (int) floor( ((double) cake_cntx->alpha_n*p*mc_ret) / Wout);
	nc_ret = Hc;
	// nc_ret -= (nc_ret % cake_cntx->nr);
	// nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;
	nc_ret = nc_ret == 0 ? 1 : nc_ret;

	// L2 size <= (4.0*(Cm*Wf*Hf*Ck + Hc*Win*Ck + Hc*Wout*Cm))
	Ck = (int) ceil( ((((double) cake_cntx->L2) / 4.0) - nc_ret*Wout*mc_ret) / (mc_ret*Wf*Hf + nc_ret*Win) );

	if(Ck > Cin) {
		Ck = Cin;
	} else if(Ck < 0) {
		Ck = 128;
	}
	// Ck += Cin % Ck;
	Ck = 2;

	blk_ret->m_c = mc_ret;
	blk_ret->k_c = Ck;
	blk_ret->n_c = nc_ret;
	blk_ret->sch = sch;

	printf("Cm = %d, Ck = %d, Hc = %d, Cout = %d, Cin = %d\n", mc_ret, Ck, nc_ret, Cout, Cin);
	// exit(1);

	// } else {
		// use default cake gemm block sizing
	// }


	return blk_ret;
}



void init_block_dims_conv(int Wf, int Hf, int Win, int Hin, 
	int Wout, int Hout, int Cin, int Cout, int s, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, 
	char* argv[], float density, float type_size) {

	int m_r = cake_cntx->mr;
	int n_r = cake_cntx->nr;

	cache_dims_t* cache_dims = get_cache_dims_conv(Wf, Hf, Win, Hin, Wout, Hout, 
		Cin, Cout, s, p, cake_cntx, sch, argv, density, type_size);

    x->m_c = cache_dims->m_c;
	x->k_c = cache_dims->k_c;
    x->n_c = cache_dims->n_c;
    x->sch = cache_dims->sch;
    free(cache_dims);
   
    int N = Hout;
    int M = Cout;
    int K = Cin;

	switch(x->sch) {

		case KMN: {

			x->k_pad = (K % x->k_c) ? 1 : 0; 
			x->n_pad = (N % (x->n_c)) ? 1 : 0; 
			x->m_pad = (M % (p*x->m_c)) ? 1 : 0; 

			x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r) ;
			int mr_per_core = (int) ceil( ((double) x->mr_rem) / p );

			if(mr_per_core) 
				x->p_l = (int) ceil( ((double) x->mr_rem) / mr_per_core);
			else
				x->p_l = 0;

			// x->nr_rem = (int) ceil( ((double) (N % (x->n_c * Wout)) / n_r)) ;
			// x->n_c1 = x->nr_rem * n_r;
			// x->nr_rem = (int) ceil( ((double) (N % (x->n_c))) ;
			x->n_c1 = N % (x->n_c);

			x->m_c1 = mr_per_core * m_r;
			x->m_c1_last_core = (mr_per_core - (x->p_l*mr_per_core - x->mr_rem)) * m_r;
			x->k_c1 = K % x->k_c;

			//number of CB blocks in the M, N, and K dims
			x->Mb = (M / (p*x->m_c)) + x->m_pad;
			x->Nb = (N / (x->n_c)) + x->n_pad;
			x->Kb = (K / x->k_c) + x->k_pad;

			x->M_padded = (m_r*x->mr_rem + (M / (p*x->m_c))*p*x->m_c);
			x->N_padded = (N - (N % x->n_c)) + x->n_c1;


			printf("m_c1 = %d, n_c1 = %d, k_c1 = %d, "
				"M_padded = %d, N_padded = %d, "
				"Mb = %d, Nb = %d, Kb = %d, "
				"m_pad = %d, n_pad = %d, k_pad = %d, "
				"mr_rem = %d, m_c1_last_core = %d, p_l = %d\n", 
				x->m_c1, x->n_c1, x->k_c1, x->M_padded, x->N_padded,
				x->Mb, x->Nb, x->Kb, x->m_pad, x->n_pad, x->k_pad, 
				x->mr_rem, x->m_c1_last_core, x->p_l);

	// exit(1);

			break;
		}


		case MKN: {

			break;
		}


		case NKM: {


			break;
		}


		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}
}








// pack the entire matrix A into a single contiguous cache-aligned buffer
double pack_filters(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr;

	int m_c = x->m_c, k_c = x->k_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
	int Mb = x->Mb, Kb = x->Kb;

	struct timespec start, end;
	double diff_t;

	int m, k, A_offset = 0, A_p_offset = 0;
	int m_cb, k_c_t, p_used, core;

	k_c *= 9;
	k_c1 *= 9;


   clock_gettime(CLOCK_REALTIME, &start);

   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = m_r*mr_rem ; 
      } else {
         p_used = p;
         m_cb = p_used*m_c;
      }

      for(k = 0; k < Kb; k++) {
         
         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }

         A_offset = m*p*m_c*K + k*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;
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

            pack_ob_A_single_buf(&A[A_offset + core*m_c_x*K], &A_p[A_p_offset + core*m_c_x*k_c_t], 
               M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
         }

         A_p_offset += m_cb*k_c_t;
      }
   }

     clock_gettime(CLOCK_REALTIME, &end);
     long seconds = end.tv_sec - start.tv_sec;
     long nanoseconds = end.tv_nsec - start.tv_nsec;
     diff_t = seconds + nanoseconds*1e-9;

     return diff_t;
}




void schedule_KMN_conv(float* F, float* In, float* Out, int Wf, int Hf, 
	int Win, int Hin, int Wout, int Hout, int Cin, int Cout, int s, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;
	int m_map = cake_cntx->m_map, n_map = cake_cntx->n_map;

	int Cm = x->m_c, Ck = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int m_c1_last_core = x->m_c1_last_core;
	int mr_rem = x->mr_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;

	int m, k, n, Hc;
	int m_cb, n_c_t, p_used, core;
	// Hout = (Hin - (Hf - s)) / s;
	// Wout = (Win - (Wf - s)) / s;


	int K = Cin*Wf*Hf;
	int N_in = Win*Hin;	
	int N_out = Wout*Hout;


	// printf("Nb = %d, Mb = %d, Kb = %d\n", Nb, Mb, Kb);
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
				m_cb = p_used*Cm;
			}

			#pragma omp parallel for private(core, k)
			for(core = 0; core < p_used; core++) {

				// These vars must be private to thread, 
				// otherwise out of bounds memory access possible
				int m_c_t, m_c_x, k_c_t, n_reg, m_reg;
				int ho, wf, hf;

				if((m == Mb - 1) && m_pad) {
					m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
					m_c_x = m_c1;
				} else {
					m_c_t = Cm;
					m_c_x = Cm; 
				}

				for(ho = 0; ho < n_c_t; ho++) {
					for(hf = 0; hf < Hf; hf++) {
						for(wf = 0; wf < Wf; wf++) {
							for(k = 0; k < Kb; k++) {
							
								k_c_t = Ck; 
								if((k == Kb - 1) && k_pad) {
									k_c_t = k_c1;
								}
				// for(ho = 0; ho < n_c_t; ho++) {

								// filter/input/output feature map index within CB block
								int f_ind = m*p*Cm*K + k*m_cb*Ck*Wf*Hf + core*m_c_x*k_c_t*Wf*Hf + m_r*k_c_t*(wf + hf*Wf); 
								int in_ind = k*Ck*Hin*Win + n*n_c*Win + ho*Win + hf*Win + wf; 
								int out_ind = m*p*Cm*Hout*Wout + n*n_c*Wout + core*m_c_x*Hout*Wout + ho*Wout;
					// printf("f_ind %d, in_ind %d, out_ind %d \n", f_ind, in_ind, out_ind);

								for(n_reg = 0; n_reg < (Wout / n_r); n_reg++) {
									for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {

										cake_conv_haswell_6x16(&F[f_ind + m_reg*m_r*k_c_t*Wf*Hf], 
												&In[in_ind + n_reg*n_r], 
												&Out[out_ind + n_reg*n_r + m_reg*m_r*Hout*Wout], 
												N_in, N_out, k_c_t);

									}
								}	
							}
						}
					}
				}
			}
		}
	}
}


	// 	for(m = m_start; m != m_end; m += m_inc) {

			// if((m == Mb - 1) && m_pad) {
			// 	p_used = p_l;
			// 	m_cb = m_r*mr_rem ; //M % (p*m_c);
			// } else {
			// 	p_used = p;
			// 	m_cb = p_used*m_c;
			// }

	// 		// pragma omp here (i_c loop)
	// 		#pragma omp parallel for private(core,k)
	// 		for(core = 0; core < p_used; core++) {

	// 			// These vars must be private to thread, 
	// 			// otherwise out of bounds memory access possible
	// 			int m_c_t, m_c_x, k_c_t, n_reg, m_reg;

				// if((m == Mb - 1) && m_pad) {
				// 	m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
				// 	m_c_x = m_c1;
				// } else {
				// 	m_c_t = m_c;
				// 	m_c_x = m_c; 
				// }


	// 			// pragma omp also here possible (j_r loop)
	// 			for(k = k_start; k != k_end; k += k_inc) {
					
	// 				k_c_t = k_c; 
	// 				if((k == Kb - 1) && k_pad) {
	// 					k_c_t = k_c1;
	// 				}

	// 				int a_ind = m*p*m_c*K + k*m_cb*k_c + core*m_c_x*k_c_t;
	// 				int b_ind = n*K*n_c + k*k_c*n_c_t;
	// 				int c_ind = n*M_padded*n_c + m*p*m_c*n_c_t + core*m_c_x*n_c_t;

	// 				for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
	// 					for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

	// 						kernel_map[m_map][n_map](&A_p[a_ind + m_reg*m_r*k_c_t], 
	// 										&B_p[b_ind + n_reg*k_c_t*n_r], 
	// 										&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
	// 										m_r, n_r, k_c_t);
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// }


double cake_sconv(float* A, float* B, float* C, int Wf, int Hf, 
	int Win, int Hin, int Wout, int Hout, int Cin, int Cout, int s, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, enum sched sch) {


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);


	int A_sz, B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;


	clock_gettime(CLOCK_REALTIME, &start1);

	sch = KMN;
	init_block_dims_conv(Wf, Hf, Win, Hin, Wout, Hout, 
		Cin, Cout, s, p, x, cake_cntx, sch, argv, 0);
	// sch = x->sch;


	if(packedA) {
		A_p = A;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(Cout, Cin*Wf*Hf, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &A_p, 64, A_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_filters(A, A_p, Cout, Cin*Wf*Hf, p, x, cake_cntx);
		// print_mat(A_p, 6, Cin*Wf*Hf);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		printf("A pack time: %f \n", diff_t ); 
	}

	// B_sz = Hin*Win * Cin * sizeof( float );
	// if(posix_memalign((void**) &B_p, 64, B_sz)) {
	// 	printf("posix memalign error\n");
	// 	exit(1);
	// }

	// for(int i = 0; i < B_sz / (sizeof(float)); i++) {
	// 	B_p[i] = B[i];
	// }

	clock_gettime(CLOCK_REALTIME, &start);

	schedule_KMN_conv(A_p, B, C, Wf, Hf, Win, Hin, Wout, Hout, 
		Cin, Cout, s, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	// if(DEBUG) printf("CONV time: %f \n", diff_t); 	// exit(1);
	printf("CONV time: %f \n", diff_t); 	// exit(1);

	times = diff_t;

	if(!packedA) free(A_p);
	free(x);


	return times;
}






// randomized double precision matrices in range [-1,1]
void rand_init1(float* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		// mat[i] =  (float) rand() / ((float) RAND_MAX)*2.0 - 1.0;
	}	
}


int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, Cin, Cout, Wf, Hf, Hin, Win, Hout, Wout, s, p;
	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t;

	float *F, *In, *Out;
	// Cin = atoi(argv[1]);
	// Cout = atoi(argv[2]);
	// Wf = atoi(argv[3]);
	// Hf = atoi(argv[4]);
	// Hin = atoi(argv[5]);
	// Win = atoi(argv[6]);

	// p = atoi(argv[7]);

	// single out channel, single in channel
	Cin = 3;
	Cout = 1;
	Wf = 3;
	Hf = 3;
	Hin = 18;
	Win = 18;
	Hout = 16;
	Wout = 16;
	s = 1;
	p = 1;

	// Cin = 512;
	// Cout = 512;
	// Wf = 3;
	// Hf = 3;
	// Hin = 33;
	// Win = 33;
	// Hout = 32;
	// Wout = 32;
	// s = 1;
	// p = 10;

	cake_cntx_t* cake_cntx = cake_query_cntx();

	// printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);
	F = (float*) malloc(Wf*Hf*Cin * Cout * sizeof( float ));
	In = (float*) malloc(Hin*Win * Cin * sizeof( float ));

	// initialize A and B
    // srand(time(NULL));
	srand(1);
	rand_init(In, Cin, Hin*Win);
	rand_init(F, Cout, Wf*Hf*Cin);


	
	int ntrials = 1;
	double ret = 0;
	clock_gettime(CLOCK_REALTIME, &start);

	for(int i = 0; i < ntrials; i++) {
		Out = (float*) calloc(Hout*Wout * Cout , sizeof( float ));
		ret += cake_sconv(F, In, Out, Wf, Hf, Win, Hin, 
			Wout, Hout, Cin, Cout, s, p, cake_cntx);
	}
	
    clock_gettime(CLOCK_REALTIME, &end);
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("conv time: %f \n", ret / ntrials); 

	// print_mat(F, Cout, Cin*Wf*Hf);
	// print_filters(F, Wf, Hf, Cin, Cout);
	// print_feature_map(In, Win, Hin, Cin);
	print_feature_map(Out, Wout, Hout, Cout);

	// free(Out);



	// free(F);
	// free(In);






	M = Cout;
	K = Wf*Hf*Cin;
	N = Hout*Wout;

	printf("\n\nM = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

	float* B = im_2_col(In, Wf, Hf, Win, Hin, Wout, Hout, Cin, Cout, s);
	// print_mat(B, K, N);
	// float* A = (float*) calloc(M * K , sizeof( float ));
	// float* B = (float*) calloc(K * N , sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));
	
	// rand_init(A, M, K);
	// rand_init(B, K, N);
	float *A_p, *B_p;
	enum sched sch = KMN;
	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, sch, argv, 0);
	int A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);	
	if(posix_memalign((void**) &A_p, 64, A_sz)) {
		printf("posix memalign error\n");
		exit(1);
	}
	pack_A(F, A_p, M, K, p, x, cake_cntx, sch);


    int B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
	if(posix_memalign((void**) &B_p, 64, B_sz)) {
		printf("posix memalign error\n");
		exit(1);
	}
	
	pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

	ret = 0;
    clock_gettime(CLOCK_REALTIME, &start);

	for(int i = 0; i < ntrials; i++) {
		ret += cake_sgemm(A_p, B_p, C, M, N, K, p, cake_cntx, NULL, 1, 1, 1, 0, KMN);
	}

    clock_gettime(CLOCK_REALTIME, &end);
     seconds = end.tv_sec - start.tv_sec;
     nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sgemm time: %f \n", ret / ntrials); 

	print_mat(C, M, N);
	conv_gemm_checker(Out, C, M, N);

	free(F);
	free(In);
	free(Out);
	free(B);
	free(C);	
	free(cake_cntx);
	
	return 0;
}




void print_filters(float* F, int Wf, int Hf, int Cin, int Cout) {

	for(int n = 0; n < Cout; n++) {
		printf("filter %d\n\n", n);
		for(int k = 0; k < Cin; k++) {
			printf("channel %d\n", k);
			for(int h = 0; h < Hf; h++) {
				for(int w = 0; w < Wf; w++) {
					printf("%f, ", F[n*Cin*Wf*Hf + (w + h*Wf)*Cin + k]);
				}
				// printf("\n");
			}
			printf("\n\n");
		}
		printf("\n\n\n");
	}


}



void print_feature_map(float* In, int Win, int Hin, int Cin) {

	for(int n = 0; n < Cin; n++) {
		printf("channel %d\n", n);
		for(int h = 0; h < Hin; h++) {
			for(int w = 0; w < Win; w++) {
				printf("%f ", In[n*Win*Hin + h*Win + w]);
			}
			printf("\n");
		}
	}
	printf("\n\n");
	
}



float* im_2_col(float* In, int Wf, int Hf, int Win, int Hin, 
	int Wout, int Hout, int Cin, int Cout, int s) {

	float* mat = (float*) malloc(Cin*Wf*Hf * Win*Hin * sizeof(float));
	float* tmp = (float*) malloc(Cin*Wf*Hf * Win*Hin * sizeof(float));
	int ind = 0;

	int N = Hout*Wout;
	int K = Wf*Hf*Cin;

	for(int h1 = 0; h1 < Hout; h1++) {
		for(int w1 = 0; w1 < Wout; w1++) {

			for(int h = 0; h < Hf; h++) {
				for(int  w = 0; w < Wf; w++) {
					for(int k = 0; k < Cin; k++) {
						tmp[ind] = In[k*Hin*Win + w1 + h1*Win + w + h*Win];
						ind++;
					}
				}
			}
		}
	}

	// transpose 
	ind = 0;
	for(int k = 0; k < K; k++) {
		for(int n = 0; n < N; n++) {
			mat[ind] = tmp[n*K + k];
			ind++;
		}
	}

	free(tmp);
	return mat;
}


void conv_gemm_checker(float* conv, float* gemm, int M, int N) {

	int WRONG = 0, CORRECT = 0;
    float eps = 1e-3; // machine precision level

	for(int i = 0; i < M*N; i++) {
		if( fabs(gemm[i] - conv[i]) > eps) {
			WRONG++;
		} else {
			CORRECT++;
		}
        if(CHECK_PRINT) printf("%f\t%f\n", gemm[i], conv[i]);
	}

	if(WRONG)
		printf("WRONG = %d , CORRECT = %d\n", WRONG, CORRECT);
	 else
		printf("CORRECT!\n");
}
