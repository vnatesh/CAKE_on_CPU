

//-----------------------------Dense Kernels--------------------------------------

// assumes A, B, and C are all packed
void cake_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k) {

	float32x4_t a1, a2, a3, a4, b1, b2, b3;
	float32x4_t c[8*3];

	// load 6x16 tile of C into 12 AVX2 registers
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
	c[19]= vld1q_f32(C + 76);
	c[20]= vld1q_f32(C + 80);
	c[21]= vld1q_f32(C + 84);
	c[22]= vld1q_f32(C + 88);
	c[23]= vld1q_f32(C + 92);


	int rem = k % 4;
	k -= rem;

	// 8x12 outer-product
	for(int kk = 0; kk < k; kk += 4) { 

		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[0] =  vfmaq_f32(c[0],  b1, a1);
		c[1] =  vfmaq_f32(c[1],  b2, a1);
		c[2] =  vfmaq_f32(c[2],  b3, a1);
		c[3] =  vfmaq_f32(c[3],  b1, a2);
		c[4] =  vfmaq_f32(c[4],  b2, a2);
		c[5] =  vfmaq_f32(c[5],  b3, a2);
		c[6] =  vfmaq_f32(c[6],  b1, a3);
		c[7] =  vfmaq_f32(c[7],  b2, a3);
		c[8] =  vfmaq_f32(c[8],  b3, a3);
		c[9] =  vfmaq_f32(c[9],  b1, a4);
		c[10] = vfmaq_f32(c[10], b2, a4);
		c[11] = vfmaq_f32(c[11], b3, a4);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[12] =  vfmaq_f32(c[12],  b1, a1);
		c[13] =  vfmaq_f32(c[13],  b2, a1);
		c[14] =  vfmaq_f32(c[14],  b3, a1);
		c[15] =  vfmaq_f32(c[15],  b1, a2);
		c[16] =  vfmaq_f32(c[16],  b2, a2);
		c[17] =  vfmaq_f32(c[17],  b3, a2);
		c[18] =  vfmaq_f32(c[18],  b1, a3);
		c[19] =  vfmaq_f32(c[19],  b2, a3);
		c[20] =  vfmaq_f32(c[20],  b3, a3);
		c[21] =  vfmaq_f32(c[21],  b1, a4);
		c[22] = vfmaq_f32(c[22], b2, a4);
		c[23] = vfmaq_f32(c[23], b3, a4);

		B += n;




		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[0] =  vfmaq_f32(c[0],  b1, a1);
		c[1] =  vfmaq_f32(c[1],  b2, a1);
		c[2] =  vfmaq_f32(c[2],  b3, a1);
		c[3] =  vfmaq_f32(c[3],  b1, a2);
		c[4] =  vfmaq_f32(c[4],  b2, a2);
		c[5] =  vfmaq_f32(c[5],  b3, a2);
		c[6] =  vfmaq_f32(c[6],  b1, a3);
		c[7] =  vfmaq_f32(c[7],  b2, a3);
		c[8] =  vfmaq_f32(c[8],  b3, a3);
		c[9] =  vfmaq_f32(c[9],  b1, a4);
		c[10] = vfmaq_f32(c[10], b2, a4);
		c[11] = vfmaq_f32(c[11], b3, a4);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[12] =  vfmaq_f32(c[12],  b1, a1);
		c[13] =  vfmaq_f32(c[13],  b2, a1);
		c[14] =  vfmaq_f32(c[14],  b3, a1);
		c[15] =  vfmaq_f32(c[15],  b1, a2);
		c[16] =  vfmaq_f32(c[16],  b2, a2);
		c[17] =  vfmaq_f32(c[17],  b3, a2);
		c[18] =  vfmaq_f32(c[18],  b1, a3);
		c[19] =  vfmaq_f32(c[19],  b2, a3);
		c[20] =  vfmaq_f32(c[20],  b3, a3);
		c[21] =  vfmaq_f32(c[21],  b1, a4);
		c[22] = vfmaq_f32(c[22], b2, a4);
		c[23] = vfmaq_f32(c[23], b3, a4);

		B += n;




		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[0] =  vfmaq_f32(c[0],  b1, a1);
		c[1] =  vfmaq_f32(c[1],  b2, a1);
		c[2] =  vfmaq_f32(c[2],  b3, a1);
		c[3] =  vfmaq_f32(c[3],  b1, a2);
		c[4] =  vfmaq_f32(c[4],  b2, a2);
		c[5] =  vfmaq_f32(c[5],  b3, a2);
		c[6] =  vfmaq_f32(c[6],  b1, a3);
		c[7] =  vfmaq_f32(c[7],  b2, a3);
		c[8] =  vfmaq_f32(c[8],  b3, a3);
		c[9] =  vfmaq_f32(c[9],  b1, a4);
		c[10] = vfmaq_f32(c[10], b2, a4);
		c[11] = vfmaq_f32(c[11], b3, a4);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[12] =  vfmaq_f32(c[12],  b1, a1);
		c[13] =  vfmaq_f32(c[13],  b2, a1);
		c[14] =  vfmaq_f32(c[14],  b3, a1);
		c[15] =  vfmaq_f32(c[15],  b1, a2);
		c[16] =  vfmaq_f32(c[16],  b2, a2);
		c[17] =  vfmaq_f32(c[17],  b3, a2);
		c[18] =  vfmaq_f32(c[18],  b1, a3);
		c[19] =  vfmaq_f32(c[19],  b2, a3);
		c[20] =  vfmaq_f32(c[20],  b3, a3);
		c[21] =  vfmaq_f32(c[21],  b1, a4);
		c[22] = vfmaq_f32(c[22], b2, a4);
		c[23] = vfmaq_f32(c[23], b3, a4);

		B += n;




		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[0] =  vfmaq_f32(c[0],  b1, a1);
		c[1] =  vfmaq_f32(c[1],  b2, a1);
		c[2] =  vfmaq_f32(c[2],  b3, a1);
		c[3] =  vfmaq_f32(c[3],  b1, a2);
		c[4] =  vfmaq_f32(c[4],  b2, a2);
		c[5] =  vfmaq_f32(c[5],  b3, a2);
		c[6] =  vfmaq_f32(c[6],  b1, a3);
		c[7] =  vfmaq_f32(c[7],  b2, a3);
		c[8] =  vfmaq_f32(c[8],  b3, a3);
		c[9] =  vfmaq_f32(c[9],  b1, a4);
		c[10] = vfmaq_f32(c[10], b2, a4);
		c[11] = vfmaq_f32(c[11], b3, a4);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[12] =  vfmaq_f32(c[12],  b1, a1);
		c[13] =  vfmaq_f32(c[13],  b2, a1);
		c[14] =  vfmaq_f32(c[14],  b3, a1);
		c[15] =  vfmaq_f32(c[15],  b1, a2);
		c[16] =  vfmaq_f32(c[16],  b2, a2);
		c[17] =  vfmaq_f32(c[17],  b3, a2);
		c[18] =  vfmaq_f32(c[18],  b1, a3);
		c[19] =  vfmaq_f32(c[19],  b2, a3);
		c[20] =  vfmaq_f32(c[20],  b3, a3);
		c[21] =  vfmaq_f32(c[21],  b1, a4);
		c[22] = vfmaq_f32(c[22], b2, a4);
		c[23] = vfmaq_f32(c[23], b3, a4);

		B += n;

	}
	
	// lefotver k elements
	for(int kk = 0; kk < rem; kk++) {

		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[0] =  vfmaq_f32(c[0],  b1, a1);
		c[1] =  vfmaq_f32(c[1],  b2, a1);
		c[2] =  vfmaq_f32(c[2],  b3, a1);
		c[3] =  vfmaq_f32(c[3],  b1, a2);
		c[4] =  vfmaq_f32(c[4],  b2, a2);
		c[5] =  vfmaq_f32(c[5],  b3, a2);
		c[6] =  vfmaq_f32(c[6],  b1, a3);
		c[7] =  vfmaq_f32(c[7],  b2, a3);
		c[8] =  vfmaq_f32(c[8],  b3, a3);
		c[9] =  vfmaq_f32(c[9],  b1, a4);
		c[10] = vfmaq_f32(c[10], b2, a4);
		c[11] = vfmaq_f32(c[11], b3, a4);

		a1 = vld1q_dup_f32(A++);
		a2 = vld1q_dup_f32(A++);
		a3 = vld1q_dup_f32(A++);
		a4 = vld1q_dup_f32(A++);

		c[12] =  vfmaq_f32(c[12],  b1, a1);
		c[13] =  vfmaq_f32(c[13],  b2, a1);
		c[14] =  vfmaq_f32(c[14],  b3, a1);
		c[15] =  vfmaq_f32(c[15],  b1, a2);
		c[16] =  vfmaq_f32(c[16],  b2, a2);
		c[17] =  vfmaq_f32(c[17],  b3, a2);
		c[18] =  vfmaq_f32(c[18],  b1, a3);
		c[19] =  vfmaq_f32(c[19],  b2, a3);
		c[20] =  vfmaq_f32(c[20],  b3, a3);
		c[21] =  vfmaq_f32(c[21],  b1, a4);
		c[22] = vfmaq_f32(c[22], b2, a4);
		c[23] = vfmaq_f32(c[23], b3, a4);

		B += n;

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


