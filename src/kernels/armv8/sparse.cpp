#include "cake.h"






void cake_sp_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* k_inds, int* loc_m) {

	int m_cnt, k_ind;
	float32x4_t a, b1, b2, b3;
	float32x4_t c[8*3];

	// load 8x12 tile of C into 24 neon simd registers
	c[0] = vld1q_f32(C);
	c[1] = vld1q_f32(C + 4);
	c[2] = vld1q_f32(C + 8);
	c[3] = vld1q_f32(C + 12);
	c[4] = vld1q_f32(C + 16);
	c[5] = vld1q_f32(C + 20);
	c[6] = vld1q_f32(C + 24);
	c[7] = vld1q_f32(C + 28);
	c[8] = vld1q_f32(C + 32);
	c[9] = vld1q_f32(C + 36);
	c[10] = vld1q_f32(C + 40);
	c[11] = vld1q_f32(C + 44);
	c[12] = vld1q_f32(C + 48);
	c[13] = vld1q_f32(C + 52);
	c[14] = vld1q_f32(C + 56);
	c[15] = vld1q_f32(C + 60);
	c[16] = vld1q_f32(C + 64);
	c[17] = vld1q_f32(C + 68);
	c[18] = vld1q_f32(C + 72);
	c[19] = vld1q_f32(C + 76);
	c[20] = vld1q_f32(C + 80);
	c[21] = vld1q_f32(C + 84);
	c[22] = vld1q_f32(C + 88);
	c[23] = vld1q_f32(C + 92);


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

		b1 = vld1q_f32(B + k_ind);
		b2 = vld1q_f32(B + k_ind + 4);
		b3 = vld1q_f32(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = vld1q_dup_f32(A++);
			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+1];
		k_ind = n*k_inds[kk+1];

		b1 = vld1q_f32(B + k_ind);
		b2 = vld1q_f32(B + k_ind + 4);
		b3 = vld1q_f32(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = vld1q_dup_f32(A++);
			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+2];
		k_ind = n*k_inds[kk+2];

		b1 = vld1q_f32(B + k_ind);
		b2 = vld1q_f32(B + k_ind + 4);
		b3 = vld1q_f32(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = vld1q_dup_f32(A++);
			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
			loc_m++;
		}


		m_cnt = nnz_outer[kk+3];
		k_ind = n*k_inds[kk+3];

		b1 = vld1q_f32(B + k_ind);
		b2 = vld1q_f32(B + k_ind + 4);
		b3 = vld1q_f32(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = vld1q_dup_f32(A++);
			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
			loc_m++;
		}
	}


	for(int kk = 0; kk < rem; kk++) { 

		m_cnt = nnz_outer[k + kk];
		k_ind = n*k_inds[k + kk];

		b1 = vld1q_f32(B + k_ind);
		b2 = vld1q_f32(B + k_ind + 4);
		b3 = vld1q_f32(B + k_ind + 8);

		for(int j = 0; j < m_cnt; j++) {
			a = vld1q_dup_f32(A++);
			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
			loc_m++;
		}
	}


	vst1q_f32(C     , c[0] );
	vst1q_f32(C + 4 , c[1] );
	vst1q_f32(C + 8 , c[2] );
	vst1q_f32(C + 12, c[3] );
	vst1q_f32(C + 16, c[4] );
	vst1q_f32(C + 20, c[5] );
	vst1q_f32(C + 24, c[6] );
	vst1q_f32(C + 28, c[7] );
	vst1q_f32(C + 32, c[8] );
	vst1q_f32(C + 36, c[9] );
	vst1q_f32(C + 40, c[10]);
	vst1q_f32(C + 44, c[11]);
	vst1q_f32(C + 48, c[12]);
	vst1q_f32(C + 52, c[13]);
	vst1q_f32(C + 56, c[14]);
	vst1q_f32(C + 60, c[15]);
	vst1q_f32(C + 64, c[16]);
	vst1q_f32(C + 68, c[17]);
	vst1q_f32(C + 72, c[18]);
	vst1q_f32(C + 76, c[19]);
	vst1q_f32(C + 80, c[20]);
	vst1q_f32(C + 84, c[21]);
	vst1q_f32(C + 88, c[22]);
	vst1q_f32(C + 92, c[23]);

}







// // sparse kernel without density-based reordering
// void cake_sp_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* k_inds, int* loc_m) {

// 	int m_cnt, k_ind;
// 	float32x4_t a, b1, b2, b3;
// 	float32x4_t c[8*3];

// 	// load 8x12 tile of C into 24 neon simd registers
// 	c[0] = vld1q_f32(C);
// 	c[1] = vld1q_f32(C + 4);
// 	c[2] = vld1q_f32(C + 8);
// 	c[3] = vld1q_f32(C + 12);
// 	c[4] = vld1q_f32(C + 16);
// 	c[5] = vld1q_f32(C + 20);
// 	c[6] = vld1q_f32(C + 24);
// 	c[7] = vld1q_f32(C + 28);
// 	c[8] = vld1q_f32(C + 32);
// 	c[9] = vld1q_f32(C + 36);
// 	c[10] = vld1q_f32(C + 40);
// 	c[11] = vld1q_f32(C + 44);
// 	c[12] = vld1q_f32(C + 48);
// 	c[13] = vld1q_f32(C + 52);
// 	c[14] = vld1q_f32(C + 56);
// 	c[15] = vld1q_f32(C + 60);
// 	c[16] = vld1q_f32(C + 64);
// 	c[17] = vld1q_f32(C + 68);
// 	c[18] = vld1q_f32(C + 72);
// 	c[19] = vld1q_f32(C + 76);
// 	c[20] = vld1q_f32(C + 80);
// 	c[21] = vld1q_f32(C + 84);
// 	c[22] = vld1q_f32(C + 88);
// 	c[23] = vld1q_f32(C + 92);


// 	int rem = k % 4;
// 	k -= rem;
// 	// float* B_ptr = &B[0];

// 	for(int kk = 0; kk < k; kk += 4) { 

// 		m_cnt = nnz_outer[kk];
// 		b1 = vld1q_f32(B);
// 		b2 = vld1q_f32(B + 4);
// 		b3 = vld1q_f32(B + 8);

// 		for(int j = 0; j < m_cnt; j++) {
// 			a = vld1q_dup_f32(A++);
// 			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
// 			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
// 			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
// 			loc_m++;
// 		}

// 		B += n;


// 		m_cnt = nnz_outer[kk+1];
// 		b1 = vld1q_f32(B);
// 		b2 = vld1q_f32(B + 4);
// 		b3 = vld1q_f32(B + 8);

// 		for(int j = 0; j < m_cnt; j++) {
// 			a = vld1q_dup_f32(A++);
// 			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
// 			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
// 			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
// 			loc_m++;
// 		}

// 		B += n;


// 		m_cnt = nnz_outer[kk+2];
// 		b1 = vld1q_f32(B);
// 		b2 = vld1q_f32(B + 4);
// 		b3 = vld1q_f32(B + 8);

// 		for(int j = 0; j < m_cnt; j++) {
// 			a = vld1q_dup_f32(A++);
// 			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
// 			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
// 			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
// 			loc_m++;
// 		}

// 		B += n;


// 		m_cnt = nnz_outer[kk+3];
// 		b1 = vld1q_f32(B);
// 		b2 = vld1q_f32(B + 4);
// 		b3 = vld1q_f32(B + 8);

// 		for(int j = 0; j < m_cnt; j++) {
// 			a = vld1q_dup_f32(A++);
// 			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
// 			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
// 			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
// 			loc_m++;
// 		}

// 		B += n;
// 	}


// 	for(int kk = 0; kk < rem; kk++) { 

// 		m_cnt = nnz_outer[k + kk];
// 		b1 = vld1q_f32(B);
// 		b2 = vld1q_f32(B + 4);
// 		b3 = vld1q_f32(B + 8);

// 		for(int j = 0; j < m_cnt; j++) {
// 			a = vld1q_dup_f32(A++);
// 			c[*loc_m * 3] =  vfmaq_f32(c[*loc_m * 3], b1, a);
// 			c[*loc_m * 3 + 1] =  vfmaq_f32(c[*loc_m * 3 + 1], b2, a);
// 			c[*loc_m * 3 + 2] =  vfmaq_f32(c[*loc_m * 3 + 2], b3, a);
// 			loc_m++;
// 		}

// 		B += n;
// 	}


// 	vst1q_f32(C     , c[0] );
// 	vst1q_f32(C + 4 , c[1] );
// 	vst1q_f32(C + 8 , c[2] );
// 	vst1q_f32(C + 12, c[3] );
// 	vst1q_f32(C + 16, c[4] );
// 	vst1q_f32(C + 20, c[5] );
// 	vst1q_f32(C + 24, c[6] );
// 	vst1q_f32(C + 28, c[7] );
// 	vst1q_f32(C + 32, c[8] );
// 	vst1q_f32(C + 36, c[9] );
// 	vst1q_f32(C + 40, c[10]);
// 	vst1q_f32(C + 44, c[11]);
// 	vst1q_f32(C + 48, c[12]);
// 	vst1q_f32(C + 52, c[13]);
// 	vst1q_f32(C + 56, c[14]);
// 	vst1q_f32(C + 60, c[15]);
// 	vst1q_f32(C + 64, c[16]);
// 	vst1q_f32(C + 68, c[17]);
// 	vst1q_f32(C + 72, c[18]);
// 	vst1q_f32(C + 76, c[19]);
// 	vst1q_f32(C + 80, c[20]);
// 	vst1q_f32(C + 84, c[21]);
// 	vst1q_f32(C + 88, c[22]);
// 	vst1q_f32(C + 92, c[23]);

// }



