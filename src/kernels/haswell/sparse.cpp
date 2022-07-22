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

		// skip columns with 0 nonzeros
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















void cake_sp_sgemm_haswell_3x32(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* k_inds, int* loc_m) {

	int m_cnt, k_ind;
	__m256 a, b1, b2;
	__m256 c[3*4];

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

		// skip columns with 0 nonzeros
		if(!m_cnt) {
			break;
		}

		k_ind = n*k_inds[kk];	

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);
		b3 = _mm256_load_ps(B + k_ind + 16);
		b4 = _mm256_load_ps(B + k_ind + 24);

		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] 	  =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			c[*loc_m * 2 + 2] =  _mm256_fmadd_ps(a, b3, c[*loc_m * 2 + 2]);
			c[*loc_m * 2 + 3] =  _mm256_fmadd_ps(a, b4, c[*loc_m * 2 + 3]);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+1];
		k_ind = n*k_inds[kk+1];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);
		b3 = _mm256_load_ps(B + k_ind + 16);
		b4 = _mm256_load_ps(B + k_ind + 24);


		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] 	  =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			c[*loc_m * 2 + 2] =  _mm256_fmadd_ps(a, b3, c[*loc_m * 2 + 2]);
			c[*loc_m * 2 + 3] =  _mm256_fmadd_ps(a, b4, c[*loc_m * 2 + 3]);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+2];
		k_ind = n*k_inds[kk+2];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);
		b3 = _mm256_load_ps(B + k_ind + 16);
		b4 = _mm256_load_ps(B + k_ind + 24);


		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] 	  =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			c[*loc_m * 2 + 2] =  _mm256_fmadd_ps(a, b3, c[*loc_m * 2 + 2]);
			c[*loc_m * 2 + 3] =  _mm256_fmadd_ps(a, b4, c[*loc_m * 2 + 3]);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+3];
		k_ind = n*k_inds[kk+3];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);
		b3 = _mm256_load_ps(B + k_ind + 16);
		b4 = _mm256_load_ps(B + k_ind + 24);


		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] 	  =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			c[*loc_m * 2 + 2] =  _mm256_fmadd_ps(a, b3, c[*loc_m * 2 + 2]);
			c[*loc_m * 2 + 3] =  _mm256_fmadd_ps(a, b4, c[*loc_m * 2 + 3]);
			loc_m++;
		}
	}


	for(int kk = 0; kk < rem; kk++) { 

		m_cnt = nnz_outer[k + kk];
		k_ind = n*k_inds[k + kk];

		b1 = _mm256_load_ps(B + k_ind);
		b2 = _mm256_load_ps(B + k_ind + 8);
		b3 = _mm256_load_ps(B + k_ind + 16);
		b4 = _mm256_load_ps(B + k_ind + 24);


		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * 2] 	  =  _mm256_fmadd_ps(a, b1, c[*loc_m * 2]);
			c[*loc_m * 2 + 1] =  _mm256_fmadd_ps(a, b2, c[*loc_m * 2 + 1]);
			c[*loc_m * 2 + 2] =  _mm256_fmadd_ps(a, b3, c[*loc_m * 2 + 2]);
			c[*loc_m * 2 + 3] =  _mm256_fmadd_ps(a, b4, c[*loc_m * 2 + 3]);
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



// // sparse kernel without density-based reordering
// void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* k_inds,  int* loc_m) {

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






//-----------------------------------OLD----------------------------------------




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


