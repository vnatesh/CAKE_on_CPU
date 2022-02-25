#include "cake.h"



void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* loc_m) {

	__m256 a, b1, b2;
	__m256 c[6*2];
	int m_cnt;

	c[0]  = _mm256_loadu_ps(C);
	c[1]  = _mm256_loadu_ps(C + 8);
	c[2]  = _mm256_loadu_ps(C + 16);
	c[3]  = _mm256_loadu_ps(C + 24);
	c[4]  = _mm256_loadu_ps(C + 32);
	c[5]  = _mm256_loadu_ps(C + 40);
	c[6]  = _mm256_loadu_ps(C + 48);
	c[7]  = _mm256_loadu_ps(C + 56);
	c[8]  = _mm256_loadu_ps(C + 64);
	c[9]  = _mm256_loadu_ps(C + 72);
	c[10] = _mm256_loadu_ps(C + 80);
	c[11] = _mm256_loadu_ps(C + 88);


	for(int kk = 0; kk < k; kk++) { 

		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);
		m_cnt = nnz_outer[kk];
		// printf("%d ",m_cnt );
		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[(*loc_m - 1) * 2] =  _mm256_fmadd_ps(a, b1, c[(*loc_m - 1) * 2]);
			c[(*loc_m - 1)*2 + 1] =  _mm256_fmadd_ps(a, b2, c[(*loc_m - 1)*2 + 1]);
			loc_m++;
		}

		B += n;
	}

	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps((C + 8), c[1]);
	_mm256_storeu_ps((C + 16), c[2]);
	_mm256_storeu_ps((C + 24), c[3]);
	_mm256_storeu_ps((C + 32), c[4]);
	_mm256_storeu_ps((C + 40), c[5]);
	_mm256_storeu_ps((C + 48), c[6]);
	_mm256_storeu_ps((C + 56), c[7]);
	_mm256_storeu_ps((C + 64), c[8]);
	_mm256_storeu_ps((C + 72), c[9]);
	_mm256_storeu_ps((C + 80), c[10]);
	_mm256_storeu_ps((C + 88), c[11]);
}





void cake_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k) {

	__m256 a1, a2, b1, b2;
	__m256 c[6*2];

	c[0] = _mm256_loadu_ps(C);
	c[1] = _mm256_loadu_ps(C + 8);
	c[2] = _mm256_loadu_ps(C + 16);
	c[3] = _mm256_loadu_ps(C + 24);
	c[4] = _mm256_loadu_ps(C + 32);
	c[5] = _mm256_loadu_ps(C + 40);
	c[6] = _mm256_loadu_ps(C + 48);
	c[7] = _mm256_loadu_ps(C + 56);
	c[8] = _mm256_loadu_ps(C + 64);
	c[9] = _mm256_loadu_ps(C + 72);
	c[10]= _mm256_loadu_ps(C + 80);
	c[11]= _mm256_loadu_ps(C + 88);

	for(int kk = 0; kk < k; kk++) { 

		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		a1 = _mm256_broadcast_ss(A++);
		a2 = _mm256_broadcast_ss(A++);
		c[0] =  _mm256_fmadd_ps(a1, b1, c[0]);
		c[1] =  _mm256_fmadd_ps(a1, b2, c[1]);
		c[2] =  _mm256_fmadd_ps(a2, b1, c[2]);
		c[3] =  _mm256_fmadd_ps(a2, b2, c[3]);

		a1 = _mm256_broadcast_ss(A++);
		a2 = _mm256_broadcast_ss(A++);
		c[4] =  _mm256_fmadd_ps(a1, b1, c[4]);
		c[5] =  _mm256_fmadd_ps(a1, b2, c[5]);
		c[6] =  _mm256_fmadd_ps(a2, b1, c[6]);
		c[7] =  _mm256_fmadd_ps(a2, b2, c[7]);

		a1 = _mm256_broadcast_ss(A++);
		a2 = _mm256_broadcast_ss(A++);
		c[8] =  _mm256_fmadd_ps(a1, b1, c[8]);
		c[9] =  _mm256_fmadd_ps(a1, b2, c[9]);
		c[10] =  _mm256_fmadd_ps(a2, b1, c[10]);
		c[11] =  _mm256_fmadd_ps(a2, b2, c[11]);

		B += n;
	}

	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps((C + 8), c[1]);
	_mm256_storeu_ps((C + 16), c[2]);
	_mm256_storeu_ps((C + 24), c[3]);
	_mm256_storeu_ps((C + 32), c[4]);
	_mm256_storeu_ps((C + 40), c[5]);
	_mm256_storeu_ps((C + 48), c[6]);
	_mm256_storeu_ps((C + 56), c[7]);
	_mm256_storeu_ps((C + 64), c[8]);
	_mm256_storeu_ps((C + 72), c[9]);
	_mm256_storeu_ps((C + 80), c[10]);
	_mm256_storeu_ps((C + 88), c[11]);

}

// void cake_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k) {

// 	__m256 a1, a2, b1, b2, c11,c12,c21,c22,c31,c32,c41,c42,c51,c52,c61,c62;


// 	c11 = _mm256_loadu_ps(C);
// 	c12 = _mm256_loadu_ps(C + 8);
// 	c21 = _mm256_loadu_ps(C + 16);
// 	c22 = _mm256_loadu_ps(C + 24);
// 	c31 = _mm256_loadu_ps(C + 32);
// 	c32 = _mm256_loadu_ps(C + 40);
// 	c41 = _mm256_loadu_ps(C + 48);
// 	c42 = _mm256_loadu_ps(C + 56);
// 	c51 = _mm256_loadu_ps(C + 64);
// 	c52 = _mm256_loadu_ps(C + 72);
// 	c61 = _mm256_loadu_ps(C + 80);
// 	c62 = _mm256_loadu_ps(C + 88);


// 	for(int kk = 0; kk < k; kk++) {

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		a1 = _mm256_broadcast_ss(A++);
// 		a2 = _mm256_broadcast_ss(A++);
// 		c11 =  _mm256_fmadd_ps(a1, b1, c11);
// 		c12 =  _mm256_fmadd_ps(a1, b2, c12);
// 		c21 =  _mm256_fmadd_ps(a2, b1, c21);
// 		c22 =  _mm256_fmadd_ps(a2, b2, c22);

// 		a1 = _mm256_broadcast_ss(A++);
// 		a2 = _mm256_broadcast_ss(A++);
// 		c31 =  _mm256_fmadd_ps(a1, b1, c31);
// 		c32 =  _mm256_fmadd_ps(a1, b2, c32);
// 		c41 =  _mm256_fmadd_ps(a2, b1, c41);
// 		c42 =  _mm256_fmadd_ps(a2, b2, c42);

// 		a1 = _mm256_broadcast_ss(A++);
// 		a2 = _mm256_broadcast_ss(A++);
// 		c51 =  _mm256_fmadd_ps(a1, b1, c51);
// 		c52 =  _mm256_fmadd_ps(a1, b2, c52);
// 		c61 =  _mm256_fmadd_ps(a2, b1, c61);
// 		c62 =  _mm256_fmadd_ps(a2, b2, c62);

// 		B += n;
// 	}

// 	_mm256_storeu_ps(C, c11);
// 	_mm256_storeu_ps((C + 8), c12);
// 	_mm256_storeu_ps((C + 16), c21);
// 	_mm256_storeu_ps((C + 24), c22);
// 	_mm256_storeu_ps((C + 32), c31);
// 	_mm256_storeu_ps((C + 40), c32);
// 	_mm256_storeu_ps((C + 48), c41);
// 	_mm256_storeu_ps((C + 56), c42);
// 	_mm256_storeu_ps((C + 64), c51);
// 	_mm256_storeu_ps((C + 72), c52);
// 	_mm256_storeu_ps((C + 80), c61);
// 	_mm256_storeu_ps((C + 88), c62);	
// }

// void cake_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k) {

// 	__m256 a, b1, b2, c11,c12,c21,c22,c31,c32,c41,c42,c51,c52,c61,c62;


// 	c11 = _mm256_loadu_ps(C);
// 	c12 = _mm256_loadu_ps(C + 8);
// 	c21 = _mm256_loadu_ps(C + 16);
// 	c22 = _mm256_loadu_ps(C + 24);
// 	c31 = _mm256_loadu_ps(C + 32);
// 	c32 = _mm256_loadu_ps(C + 40);
// 	c41 = _mm256_loadu_ps(C + 48);
// 	c42 = _mm256_loadu_ps(C + 56);
// 	c51 = _mm256_loadu_ps(C + 64);
// 	c52 = _mm256_loadu_ps(C + 72);
// 	c61 = _mm256_loadu_ps(C + 80);
// 	c62 = _mm256_loadu_ps(C + 88);


// 	for(int kk = 0; kk < k; kk++) {

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		// 	// if(*A_ptr == 0) {
// 		// 	// 	C_ptr += n_r;
// 		// 	// } else {

// 		// 		a = _mm256_broadcast_ss(A_ptr);

// 		// 		c11 =  _mm256_fmadd_ps(a, b1, c11);
// 		// 		c12 =  _mm256_fmadd_ps(a, b2, c12);

// 		// 		C_ptr += n_r;
// 		// 	// }

// 		// 	A_ptr++;


// 		a = _mm256_broadcast_ss(A);
// 		c11 =  _mm256_fmadd_ps(a, b1, c11);
// 		c12 =  _mm256_fmadd_ps(a, b2, c12);
// 		C += n;
// 		A++;

// 		a = _mm256_broadcast_ss(A);
// 		c21 =  _mm256_fmadd_ps(a, b1, c21);
// 		c22 =  _mm256_fmadd_ps(a, b2, c22);
// 		C += n;
// 		A++;

// 		a = _mm256_broadcast_ss(A);
// 		c31 =  _mm256_fmadd_ps(a, b1, c31);
// 		c32 =  _mm256_fmadd_ps(a, b2, c32);
// 		C += n;
// 		A++;

// 		a = _mm256_broadcast_ss(A);
// 		c41 =  _mm256_fmadd_ps(a, b1, c41);
// 		c42 =  _mm256_fmadd_ps(a, b2, c42);
// 		C += n;
// 		A++;

// 		a = _mm256_broadcast_ss(A);
// 		c51 =  _mm256_fmadd_ps(a, b1, c51);
// 		c52 =  _mm256_fmadd_ps(a, b2, c52);
// 		C += n;
// 		A++;

// 		a = _mm256_broadcast_ss(A);
// 		c61 =  _mm256_fmadd_ps(a, b1, c61);
// 		c62 =  _mm256_fmadd_ps(a, b2, c62);
// 		C += n;
// 		A++;


// 		B += n;
// 		C -= m*n;
// 	}

// 	_mm256_storeu_ps(C, c11);
// 	_mm256_storeu_ps((C + 8), c12);
// 	_mm256_storeu_ps((C + 16), c21);
// 	_mm256_storeu_ps((C + 24), c22);
// 	_mm256_storeu_ps((C + 32), c31);
// 	_mm256_storeu_ps((C + 40), c32);
// 	_mm256_storeu_ps((C + 48), c41);
// 	_mm256_storeu_ps((C + 56), c42);
// 	_mm256_storeu_ps((C + 64), c51);
// 	_mm256_storeu_ps((C + 72), c52);
// 	_mm256_storeu_ps((C + 80), c61);
// 	_mm256_storeu_ps((C + 88), c62);	
// }





void cake_sgemm_haswell_7x16(float* A, float* B, float* C, int m, int n, int k) {

	__m256 a, b1, b2, c11,c12,c21,c22,c31,c32,c41,c42,c51,c52,c61,c62,c71,c72;


	c11 = _mm256_loadu_ps(C);
	c12 = _mm256_loadu_ps(C + 8);
	c21 = _mm256_loadu_ps(C + 16);
	c22 = _mm256_loadu_ps(C + 24);
	c31 = _mm256_loadu_ps(C + 32);
	c32 = _mm256_loadu_ps(C + 40);
	c41 = _mm256_loadu_ps(C + 48);
	c42 = _mm256_loadu_ps(C + 56);
	c51 = _mm256_loadu_ps(C + 64);
	c52 = _mm256_loadu_ps(C + 72);
	c61 = _mm256_loadu_ps(C + 80);
	c62 = _mm256_loadu_ps(C + 88);
	c71 = _mm256_loadu_ps(C + 96);
	c72 = _mm256_loadu_ps(C + 104);



	for(int kk = 0; kk < k; kk++) {

		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		// 	// if(*A_ptr == 0) {
		// 	// 	C_ptr += n_r;
		// 	// } else {

		// 		a = _mm256_broadcast_ss(A_ptr);

		// 		c11 =  _mm256_fmadd_ps(a, b1, c11);
		// 		c12 =  _mm256_fmadd_ps(a, b2, c12);

		// 		C_ptr += n_r;
		// 	// }

		// 	A_ptr++;


		a = _mm256_broadcast_ss(A);
		c11 =  _mm256_fmadd_ps(a, b1, c11);
		c12 =  _mm256_fmadd_ps(a, b2, c12);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c21 =  _mm256_fmadd_ps(a, b1, c21);
		c22 =  _mm256_fmadd_ps(a, b2, c22);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c31 =  _mm256_fmadd_ps(a, b1, c31);
		c32 =  _mm256_fmadd_ps(a, b2, c32);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c41 =  _mm256_fmadd_ps(a, b1, c41);
		c42 =  _mm256_fmadd_ps(a, b2, c42);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c51 =  _mm256_fmadd_ps(a, b1, c51);
		c52 =  _mm256_fmadd_ps(a, b2, c52);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c61 =  _mm256_fmadd_ps(a, b1, c61);
		c62 =  _mm256_fmadd_ps(a, b2, c62);
		C += n;
		A++;


		a = _mm256_broadcast_ss(A);
		c71 =  _mm256_fmadd_ps(a, b1, c71);
		c72 =  _mm256_fmadd_ps(a, b2, c72);
		C += n;
		A++;


		B += n;
		C -= m*n;
	}

	_mm256_storeu_ps(C, c11);
	_mm256_storeu_ps((C + 8), c12);
	_mm256_storeu_ps((C + 16), c21);
	_mm256_storeu_ps((C + 24), c22);
	_mm256_storeu_ps((C + 32), c31);
	_mm256_storeu_ps((C + 40), c32);
	_mm256_storeu_ps((C + 48), c41);
	_mm256_storeu_ps((C + 56), c42);
	_mm256_storeu_ps((C + 64), c51);
	_mm256_storeu_ps((C + 72), c52);
	_mm256_storeu_ps((C + 80), c61);
	_mm256_storeu_ps((C + 88), c62);	
	_mm256_storeu_ps((C + 96), c71);
	_mm256_storeu_ps((C + 104), c72);

}


void cake_sgemm_haswell_8x16(float* A, float* B, float* C, int m, int n, int k) {

	__m256 a, b1, b2, c11,c12,c21,c22,c31,c32,c41,c42,c51,c52,c61,c62,c71,c72,c81,c82;


	c11 = _mm256_loadu_ps(C);
	c12 = _mm256_loadu_ps(C + 8);
	c21 = _mm256_loadu_ps(C + 16);
	c22 = _mm256_loadu_ps(C + 24);
	c31 = _mm256_loadu_ps(C + 32);
	c32 = _mm256_loadu_ps(C + 40);
	c41 = _mm256_loadu_ps(C + 48);
	c42 = _mm256_loadu_ps(C + 56);
	c51 = _mm256_loadu_ps(C + 64);
	c52 = _mm256_loadu_ps(C + 72);
	c61 = _mm256_loadu_ps(C + 80);
	c62 = _mm256_loadu_ps(C + 88);
	c71 = _mm256_loadu_ps(C + 96);
	c72 = _mm256_loadu_ps(C + 104);
	c81 = _mm256_loadu_ps(C + 112);
	c82 = _mm256_loadu_ps(C + 120);


	for(int kk = 0; kk < k; kk++) {

		b1 = _mm256_load_ps(B);
		b2 = _mm256_load_ps(B + 8);

		// 	// if(*A_ptr == 0) {
		// 	// 	C_ptr += n_r;
		// 	// } else {

		// 		a = _mm256_broadcast_ss(A_ptr);

		// 		c11 =  _mm256_fmadd_ps(a, b1, c11);
		// 		c12 =  _mm256_fmadd_ps(a, b2, c12);

		// 		C_ptr += n_r;
		// 	// }

		// 	A_ptr++;


		a = _mm256_broadcast_ss(A);
		c11 =  _mm256_fmadd_ps(a, b1, c11);
		c12 =  _mm256_fmadd_ps(a, b2, c12);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c21 =  _mm256_fmadd_ps(a, b1, c21);
		c22 =  _mm256_fmadd_ps(a, b2, c22);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c31 =  _mm256_fmadd_ps(a, b1, c31);
		c32 =  _mm256_fmadd_ps(a, b2, c32);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c41 =  _mm256_fmadd_ps(a, b1, c41);
		c42 =  _mm256_fmadd_ps(a, b2, c42);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c51 =  _mm256_fmadd_ps(a, b1, c51);
		c52 =  _mm256_fmadd_ps(a, b2, c52);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c61 =  _mm256_fmadd_ps(a, b1, c61);
		c62 =  _mm256_fmadd_ps(a, b2, c62);
		C += n;
		A++;


		a = _mm256_broadcast_ss(A);
		c71 =  _mm256_fmadd_ps(a, b1, c71);
		c72 =  _mm256_fmadd_ps(a, b2, c72);
		C += n;
		A++;

		a = _mm256_broadcast_ss(A);
		c81 =  _mm256_fmadd_ps(a, b1, c81);
		c82 =  _mm256_fmadd_ps(a, b2, c82);
		C += n;
		A++;


		B += n;
		C -= m*n;
	}

	_mm256_storeu_ps(C, c11);
	_mm256_storeu_ps((C + 8), c12);
	_mm256_storeu_ps((C + 16), c21);
	_mm256_storeu_ps((C + 24), c22);
	_mm256_storeu_ps((C + 32), c31);
	_mm256_storeu_ps((C + 40), c32);
	_mm256_storeu_ps((C + 48), c41);
	_mm256_storeu_ps((C + 56), c42);
	_mm256_storeu_ps((C + 64), c51);
	_mm256_storeu_ps((C + 72), c52);
	_mm256_storeu_ps((C + 80), c61);
	_mm256_storeu_ps((C + 88), c62);	
	_mm256_storeu_ps((C + 96), c71);
	_mm256_storeu_ps((C + 104), c72);
	_mm256_storeu_ps((C + 112), c81);
	_mm256_storeu_ps((C + 120), c82);	

}

