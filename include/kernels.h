#include "common.h"




void cake_sgemm_haswell_6x16_unpacked(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);
void cake_sgemm_haswell_6x16_A_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);
void cake_sgemm_haswell_6x16_B_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);
void cake_sgemm_haswell_6x16_C_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);




// kernel helper functions
inline void cake_sgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, cake_cntx_t* cake_cntx) {


#ifdef USE_BLIS 
		// Set the scalars to use during each GEMM kernel call.
	float alpha_blis = 1.0;
	float beta_blis  = 1.0;
	// inc_t rsc, csc;
	// rsc = n_r; csc = 1;
    // rsc = 1; csc = m_r;
	auxinfo_t def_data;

	bli_sgemm_haswell_asm_6x16(m_r, n_r, k_c_t, &alpha_blis, A_p, B_p, &beta_blis, C_p, 
		(inc_t) n_r, (inc_t) 1, &def_data, (cntx_t*) cake_cntx->blis_cntx);

#elif USE_CAKE_HASWELL
	cake_sgemm_haswell_6x16(A_p, B_p, C_p, m_r, n_r, k_c_t);
#elif USE_CAKE_ARMV8
	cake_sgemm_armv8_8x12(A_p, B_p, C_p, m_r, n_r, k_c_t);
#endif

}


inline void cake_sgemm_small_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, int M, int K, int N) {

#ifdef USE_CAKE_HASWELL
	cake_sgemm_haswell_6x16_unpacked(A_p, B_p, C_p, m_r, n_r, k_c_t, M, K, N);
// #elif USE_CAKE_ARMV8
// 	cake_sgemm_haswell_6x16_unpacked(A_p, B_p, C_p, m_r, n_r, k_c_t, M, K ,N);
#endif

}

inline void cake_spgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, char* nnz_outer, int* k_inds, char* loc_m) {

#ifdef USE_CAKE_HASWELL
	cake_sp_sgemm_haswell_6x16(A_p, B_p, C_p, m_r, n_r, k_c_t, nnz_outer, k_inds, loc_m);
#elif USE_CAKE_ARMV8
	cake_sp_sgemm_armv8_8x12(A_p, B_p, C_p, m_r, n_r, k_c_t, nnz_outer, k_inds, loc_m);
#endif

}



