#include "cake.h"

//-----------------------------Sparse Kernels--------------------------------------



void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* k_inds, int* loc_m) {

	int m_cnt, k_ind;
	__m256 a, b1, b2;
	__m256 c[6*2];

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

	int rem = k % 4;
	k -= rem;

	// float* B_ptr = &B[0];

	for(int kk = 0; kk < k; kk += 4) { 

		m_cnt = nnz_outer[kk];

		if(!m_cnt) {
			break;
		}

		k_ind = n*k_inds[kk];	

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+1];
		k_ind = n*k_inds[kk+1];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+2];
		k_ind = n*k_inds[kk+2];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+3];
		k_ind = n*k_inds[kk+3];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}
	}


	for(int kk = 0; kk < rem; kk++) { 

		m_cnt = nnz_outer[k + kk];
		k_ind = n*k_inds[k + kk];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			loc_m++;
		}
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


// void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* loc_m) {

// 	int m_cnt1, m_cnt2, m_cnt3, m_cnt4;
// 	__m256 a, b1, b2;
// 	__m256 c[6*2];

// 	c[0]  = _mm256_loadu_ps(C);
// 	c[1]  = _mm256_loadu_ps(C + 8);
// 	c[2]  = _mm256_loadu_ps(C + 16);
// 	c[3]  = _mm256_loadu_ps(C + 24);
// 	c[4]  = _mm256_loadu_ps(C + 32);
// 	c[5]  = _mm256_loadu_ps(C + 40);
// 	c[6]  = _mm256_loadu_ps(C + 48);
// 	c[7]  = _mm256_loadu_ps(C + 56);
// 	c[8]  = _mm256_loadu_ps(C + 64);
// 	c[9]  = _mm256_loadu_ps(C + 72);
// 	c[10] = _mm256_loadu_ps(C + 80);
// 	c[11] = _mm256_loadu_ps(C + 88);

// 	int rem = k % 4;
// 	k -= rem;

// 	for(int kk = 0; kk < k; kk += 4) { 

// 		m_cnt1 = nnz_outer[kk];

// 		if(!m_cnt1) {
// 			break;
// 		}

// 		m_cnt2 = nnz_outer[kk+1];
// 		m_cnt3 = nnz_outer[kk+2];
// 		m_cnt4 = nnz_outer[kk+3];

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		for(int j = 0; j < m_cnt1; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;
// 		}

// 		B += n;



// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		for(int j = 0; j < m_cnt2; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;
// 		}

// 		B += n;



// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		for(int j = 0; j < m_cnt3; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;
// 		}

// 		B += n;



// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		for(int j = 0; j < m_cnt4; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;
// 		}

// 		B += n;


// 	}

// 	for(int kk = 0; kk < rem; kk++) { 

// 		m_cnt1 = nnz_outer[kk];

// 		if(!m_cnt1) {
// 			break;
// 		}

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		for(int j = 0; j < m_cnt1; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;
// 		}

// 		B += n;
// 	}


// 	_mm256_storeu_ps(C, c[0]);
// 	_mm256_storeu_ps((C + 8), c[1]);
// 	_mm256_storeu_ps((C + 16), c[2]);
// 	_mm256_storeu_ps((C + 24), c[3]);
// 	_mm256_storeu_ps((C + 32), c[4]);
// 	_mm256_storeu_ps((C + 40), c[5]);
// 	_mm256_storeu_ps((C + 48), c[6]);
// 	_mm256_storeu_ps((C + 56), c[7]);
// 	_mm256_storeu_ps((C + 64), c[8]);
// 	_mm256_storeu_ps((C + 72), c[9]);
// 	_mm256_storeu_ps((C + 80), c[10]);
// 	_mm256_storeu_ps((C + 88), c[11]);
// }



// void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* loc_m) {

// 	__m256 a, b1, b2;
// 	__m256 c[6*2];

// 	c[0]  = _mm256_loadu_ps(C);
// 	c[1]  = _mm256_loadu_ps(C + 8);
// 	c[2]  = _mm256_loadu_ps(C + 16);
// 	c[3]  = _mm256_loadu_ps(C + 24);
// 	c[4]  = _mm256_loadu_ps(C + 32);
// 	c[5]  = _mm256_loadu_ps(C + 40);
// 	c[6]  = _mm256_loadu_ps(C + 48);
// 	c[7]  = _mm256_loadu_ps(C + 56);
// 	c[8]  = _mm256_loadu_ps(C + 64);
// 	c[9]  = _mm256_loadu_ps(C + 72);
// 	c[10] = _mm256_loadu_ps(C + 80);
// 	c[11] = _mm256_loadu_ps(C + 88);


// 	for(int kk = 0; kk < k; kk++) { 

// 		const int m_cnt = nnz_outer[kk];

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		for(int j = 0; j < m_cnt; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;
// 		}

// 		B += n;
// 	}

// 	_mm256_storeu_ps(C, c[0]);
// 	_mm256_storeu_ps((C + 8), c[1]);
// 	_mm256_storeu_ps((C + 16), c[2]);
// 	_mm256_storeu_ps((C + 24), c[3]);
// 	_mm256_storeu_ps((C + 32), c[4]);
// 	_mm256_storeu_ps((C + 40), c[5]);
// 	_mm256_storeu_ps((C + 48), c[6]);
// 	_mm256_storeu_ps((C + 56), c[7]);
// 	_mm256_storeu_ps((C + 64), c[8]);
// 	_mm256_storeu_ps((C + 72), c[9]);
// 	_mm256_storeu_ps((C + 80), c[10]);
// 	_mm256_storeu_ps((C + 88), c[11]);
// }










//-----------------------------Dense Kernels--------------------------------------

// assumes A, B, and C are all packed
void cake_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k) {

	__m256 a1, a2, b1, b2;
	__m256 c[6*2];

	// load 6x16 tile of C into 12 AVX2 registers
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


	int rem = k % 4;
	k -= rem;

	// 6x16 outer-product
	for(int kk = 0; kk < k; kk += 4) { 

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
	
	// lefotver k elements
	for(int kk = 0; kk < rem; kk++) {

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


// assumes A col stored, B and C row stored
void cake_sgemm_haswell_6x16_unpacked(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N) {

	__m256 a1, a2, b1, b2;
	__m256 c[6*2];

	// load 6x16 tile of C into 12 AVX2 registers
	c[0] = _mm256_loadu_ps(C);
	c[1] = _mm256_loadu_ps(C + 8);
	c[2] = _mm256_loadu_ps(C + N);
	c[3] = _mm256_loadu_ps(C + N + 8);
	c[4] = _mm256_loadu_ps(C + 2*N);
	c[5] = _mm256_loadu_ps(C + 2*N + 8);
	c[6] = _mm256_loadu_ps(C + 3*N);
	c[7] = _mm256_loadu_ps(C + 3*N + 8);
	c[8] = _mm256_loadu_ps(C + 4*N);
	c[9] = _mm256_loadu_ps(C + 4*N + 8);
	c[10]= _mm256_loadu_ps(C + 5*N);
	c[11]= _mm256_loadu_ps(C + 5*N + 8);


	int rem = k % 4;
	k -= rem;

	// 6x16 outer-product
	for(int kk = 0; kk < k; kk += 4) { 

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
		A += M-6;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
		A += M-6;

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
		A += M-6;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
		A += M-6;

	}

	// lefotver k elements
	for(int kk = 0; kk < rem; kk++) {

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
		A += M-6;
	}
	

	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps(C + 8,c[1]);
	_mm256_storeu_ps(C + N, c[2]);
	_mm256_storeu_ps(C + N + 8, c[3]);
	_mm256_storeu_ps(C + 2*N, c[4]);
	_mm256_storeu_ps(C + 2*N + 8, c[5]);
	_mm256_storeu_ps(C + 3*N, c[6]);
	_mm256_storeu_ps(C + 3*N + 8, c[7]);
	_mm256_storeu_ps(C + 4*N, c[8]);
	_mm256_storeu_ps(C + 4*N + 8, c[9]);
	_mm256_storeu_ps(C + 5*N, c[10]);
	_mm256_storeu_ps(C + 5*N + 8, c[11]);
}






// assumes A packed, B and C row stored and unpacked
void cake_sgemm_haswell_6x16_A_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N) {

	__m256 a1, a2, b1, b2;
	__m256 c[6*2];

	// load 6x16 tile of C into 12 AVX2 registers
	c[0] = _mm256_loadu_ps(C);
	c[1] = _mm256_loadu_ps(C + 8);
	c[2] = _mm256_loadu_ps(C + N);
	c[3] = _mm256_loadu_ps(C + N + 8);
	c[4] = _mm256_loadu_ps(C + 2*N);
	c[5] = _mm256_loadu_ps(C + 2*N + 8);
	c[6] = _mm256_loadu_ps(C + 3*N);
	c[7] = _mm256_loadu_ps(C + 3*N + 8);
	c[8] = _mm256_loadu_ps(C + 4*N);
	c[9] = _mm256_loadu_ps(C + 4*N + 8);
	c[10]= _mm256_loadu_ps(C + 5*N);
	c[11]= _mm256_loadu_ps(C + 5*N + 8);


	int rem = k % 4;
	k -= rem;

	// 6x16 outer-product
	for(int kk = 0; kk < k; kk += 4) { 

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
	}

	// lefotver k elements
	for(int kk = 0; kk < rem; kk++) {

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		B += N;
	}
	

	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps(C + 8,c[1]);
	_mm256_storeu_ps(C + N, c[2]);
	_mm256_storeu_ps(C + N + 8, c[3]);
	_mm256_storeu_ps(C + 2*N, c[4]);
	_mm256_storeu_ps(C + 2*N + 8, c[5]);
	_mm256_storeu_ps(C + 3*N, c[6]);
	_mm256_storeu_ps(C + 3*N + 8, c[7]);
	_mm256_storeu_ps(C + 4*N, c[8]);
	_mm256_storeu_ps(C + 4*N + 8, c[9]);
	_mm256_storeu_ps(C + 5*N, c[10]);
	_mm256_storeu_ps(C + 5*N + 8, c[11]);
}



// assumes B packed, A col stored, C row stored 
void cake_sgemm_haswell_6x16_B_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N) {

	__m256 a1, a2, b1, b2;
	__m256 c[6*2];

	// load 6x16 tile of C into 12 AVX2 registers
	c[0] = _mm256_loadu_ps(C);
	c[1] = _mm256_loadu_ps(C + 8);
	c[2] = _mm256_loadu_ps(C + N);
	c[3] = _mm256_loadu_ps(C + N + 8);
	c[4] = _mm256_loadu_ps(C + 2*N);
	c[5] = _mm256_loadu_ps(C + 2*N + 8);
	c[6] = _mm256_loadu_ps(C + 3*N);
	c[7] = _mm256_loadu_ps(C + 3*N + 8);
	c[8] = _mm256_loadu_ps(C + 4*N);
	c[9] = _mm256_loadu_ps(C + 4*N + 8);
	c[10]= _mm256_loadu_ps(C + 5*N);
	c[11]= _mm256_loadu_ps(C + 5*N + 8);


	int rem = k % 4;
	k -= rem;

	// 6x16 outer-product
	for(int kk = 0; kk < k; kk += 4) { 

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
		A += M-6;


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
		A += M-6;


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
		A += M-6;


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
		A += M-6;
	}

	// lefotver k elements
	for(int kk = 0; kk < rem; kk++) {

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
		A += M-6;
	}
	

	_mm256_storeu_ps(C, c[0]);
	_mm256_storeu_ps(C + 8,c[1]);
	_mm256_storeu_ps(C + N, c[2]);
	_mm256_storeu_ps(C + N + 8, c[3]);
	_mm256_storeu_ps(C + 2*N, c[4]);
	_mm256_storeu_ps(C + 2*N + 8, c[5]);
	_mm256_storeu_ps(C + 3*N, c[6]);
	_mm256_storeu_ps(C + 3*N + 8, c[7]);
	_mm256_storeu_ps(C + 4*N, c[8]);
	_mm256_storeu_ps(C + 4*N + 8, c[9]);
	_mm256_storeu_ps(C + 5*N, c[10]);
	_mm256_storeu_ps(C + 5*N + 8, c[11]);
}



// C packed, A col stored, B row stored
void cake_sgemm_haswell_6x16_C_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N) {

	__m256 a1, a2, b1, b2;
	__m256 c[6*2];

	// load 6x16 tile of C into 12 AVX2 registers
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


	int rem = k % 4;
	k -= rem;

	// 6x16 outer-product
	for(int kk = 0; kk < k; kk += 4) { 

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		A += M-6;
		B += N;



		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		A += M-6;
		B += N;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		A += M-6;
		B += N;


		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		A += M-6;
		B += N;

	}
	
	// lefotver k elements
	for(int kk = 0; kk < rem; kk++) {

		b1 = _mm256_loadu_ps(B);
		b2 = _mm256_loadu_ps(B + 8);

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

		A += M-6;
		B += N;
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


//-----------------------------------OLD----------------------------------------

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





// void inner_kernel(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 		__m256 a;

// 		for(int j = 0; j < m; j++) {
// 			a = _mm256_broadcast_ss(A++);
// 			c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 			loc_m++;

// 		}

// }


// void inner_kernel1(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);
// void inner_kernel2(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);
// void inner_kernel3(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);
// void inner_kernel4(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);
// void inner_kernel5(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);
// void inner_kernel6(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);
// void inner_kernel0(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m);

// void inner_kernel0(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// }


// void inner_kernel1(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 	__m256 a;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;
// }


// void inner_kernel2(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 	__m256 a;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// }


// void inner_kernel3(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 	__m256 a;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;
// }



// void inner_kernel4(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 	__m256 a;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;
// }


// void inner_kernel5(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 	__m256 a;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;
// }


// void inner_kernel6(float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) {

// 	__m256 a;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;

// 	a = _mm256_broadcast_ss(A++);
// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 	loc_m++;
// }

	
// static void (*inner_kernel[7]) (float* A, __m256 b1, __m256 b2, __m256* c, int m, int* loc_m) =
// {
// 	inner_kernel0,
// 	inner_kernel1, 
// 	inner_kernel2, 
// 	inner_kernel3, 
// 	inner_kernel4, 
// 	inner_kernel5, 
// 	inner_kernel6
// };


// void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* loc_m) {

// 	int m_cnt1, m_cnt2, m_cnt3, m_cnt4;
// 	__m256 a, b1, b2;
// 	__m256 c[6*2];

// 	c[0]  = _mm256_loadu_ps(C);
// 	c[1]  = _mm256_loadu_ps(C + 8);
// 	c[2]  = _mm256_loadu_ps(C + 16);
// 	c[3]  = _mm256_loadu_ps(C + 24);
// 	c[4]  = _mm256_loadu_ps(C + 32);
// 	c[5]  = _mm256_loadu_ps(C + 40);
// 	c[6]  = _mm256_loadu_ps(C + 48);
// 	c[7]  = _mm256_loadu_ps(C + 56);
// 	c[8]  = _mm256_loadu_ps(C + 64);
// 	c[9]  = _mm256_loadu_ps(C + 72);
// 	c[10] = _mm256_loadu_ps(C + 80);
// 	c[11] = _mm256_loadu_ps(C + 88);

// 	int rem = k % 4;
// 	k -= rem;

// 	for(int kk = 0; kk < k; kk += 4) { 

// 		m_cnt1 = nnz_outer[kk];

		// if(!m_cnt1) {
		// 	break;
		// }

// 		m_cnt2 = nnz_outer[kk+1];
// 		m_cnt3 = nnz_outer[kk+2];
// 		m_cnt4 = nnz_outer[kk+3];

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		inner_kernel[m_cnt1](A, b1,  b2, c, m, loc_m); 

// 		loc_m += m_cnt1;
// 		A += m_cnt1;
// 		B += n;


// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		inner_kernel[m_cnt2](A, b1,  b2, c, m, loc_m); 

// 		loc_m += m_cnt2;
// 		A += m_cnt2;
// 		B += n;


// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		inner_kernel[m_cnt3](A, b1,  b2, c, m, loc_m); 

// 		loc_m += m_cnt3;
// 		A += m_cnt3;
// 		B += n;


// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		inner_kernel[m_cnt4](A, b1,  b2, c, m, loc_m); 

// 		loc_m += m_cnt4;
// 		A += m_cnt4;
// 		B += n;
// 	}

// 	for(int kk = 0; kk < rem; kk += 4) { 

// 		m_cnt1 = nnz_outer[kk];

// 		if(!m_cnt1) {
// 			break;
// 		}

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		inner_kernel[m_cnt1](A, b1,  b2, c, m, loc_m); 

// 		loc_m += m_cnt1;
// 		A += m_cnt1;
// 		B += n;
// 	}


// 	_mm256_storeu_ps(C, c[0]);
// 	_mm256_storeu_ps((C + 8), c[1]);
// 	_mm256_storeu_ps((C + 16), c[2]);
// 	_mm256_storeu_ps((C + 24), c[3]);
// 	_mm256_storeu_ps((C + 32), c[4]);
// 	_mm256_storeu_ps((C + 40), c[5]);
// 	_mm256_storeu_ps((C + 48), c[6]);
// 	_mm256_storeu_ps((C + 56), c[7]);
// 	_mm256_storeu_ps((C + 64), c[8]);
// 	_mm256_storeu_ps((C + 72), c[9]);
// 	_mm256_storeu_ps((C + 80), c[10]);
// 	_mm256_storeu_ps((C + 88), c[11]);
// }



// void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* loc_m) {

// 	// __m256* c = (__m256*) calloc(6*2 , sizeof(__m256));
// 	// __m256* b1 = (__m256*) malloc(1 * sizeof(__m256));
// 	// __m256* b2 = (__m256*) malloc(1 * sizeof(__m256));

// 	__m256 c[6*2];
// 	__m256 b1, b2;

// 	c[0]  = _mm256_loadu_ps(C);
// 	c[1]  = _mm256_loadu_ps(C + 8);
// 	c[2]  = _mm256_loadu_ps(C + 16);
// 	c[3]  = _mm256_loadu_ps(C + 24);
// 	c[4]  = _mm256_loadu_ps(C + 32);
// 	c[5]  = _mm256_loadu_ps(C + 40);
// 	c[6]  = _mm256_loadu_ps(C + 48);
// 	c[7]  = _mm256_loadu_ps(C + 56);
// 	c[8]  = _mm256_loadu_ps(C + 64);
// 	c[9]  = _mm256_loadu_ps(C + 72);
// 	c[10] = _mm256_loadu_ps(C + 80);
// 	c[11] = _mm256_loadu_ps(C + 88);


// 	for(int kk = 0; kk < k; kk++) { 
// 		const int m_cnt = nnz_outer[kk];

// 		b1 = _mm256_load_ps(B);
// 		b2 = _mm256_load_ps(B + 8);

// 		// inner_kernel(A, b1, b2, c, m_cnt, loc_m);

// 		switch(m_cnt) {
// 			case 1: {
// 				inner_kernel1(A, b1, b2, c, m_cnt, loc_m);
// 				break;
// 			}
// 			case 2: {
// 				inner_kernel2(A, b1, b2, c, m_cnt, loc_m);
// 				break;
// 			}
// 			case 3: {
// 				inner_kernel3(A, b1, b2, c, m_cnt, loc_m);
// 				break;
// 			}
// 			case 4: {
// 				inner_kernel4(A, b1, b2, c, m_cnt, loc_m);
// 				break;
// 			}
// 			case 5: {
// 				inner_kernel5(A, b1, b2, c, m_cnt, loc_m);
// 				break;
// 			}
// 			case 6: {
// 				inner_kernel6(A, b1, b2, c, m_cnt, loc_m);
// 				break;
// 			}
// 			default: {
// 				break;
// 			}
// 		}


// 		loc_m += m_cnt;
// 		A += m_cnt;

// 		B += n;

// 		// for(int j = 0; j < m_cnt; j++) {
// 		// 	a = _mm256_broadcast_ss(A++);
// 		// 	c[*loc_m * 2] =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
// 		// 	c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
// 		// 	loc_m++;
// 		// }

// 		// B += n;
// 	}

// 	_mm256_storeu_ps(C, c[0]);
// 	_mm256_storeu_ps((C + 8), c[1]);
// 	_mm256_storeu_ps((C + 16), c[2]);
// 	_mm256_storeu_ps((C + 24), c[3]);
// 	_mm256_storeu_ps((C + 32), c[4]);
// 	_mm256_storeu_ps((C + 40), c[5]);
// 	_mm256_storeu_ps((C + 48), c[6]);
// 	_mm256_storeu_ps((C + 56), c[7]);
// 	_mm256_storeu_ps((C + 64), c[8]);
// 	_mm256_storeu_ps((C + 72), c[9]);
// 	_mm256_storeu_ps((C + 80), c[10]);
// 	_mm256_storeu_ps((C + 88), c[11]);
// }


