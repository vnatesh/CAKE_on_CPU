#include "common.h"


// Dense Packing

void pack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
void pack_B(float* B, float* B_p, int K, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
double pack_A(float* A, float* A_p, int M, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
void unpack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;


void pack_test1(float* A, float* A_p, int M, int K, int p, cake_cntx_t* cake_cntx);

double pack_A_single_buf_k_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_k_first(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_k_first_blis(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_k_first_blis(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx);



double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_m_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_n_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;
void pack_B_n_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;
void pack_C_single_buf_n_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;


void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, 
				int m2, int m_c, int k_c, int m_r, bool pad);
void pack_ob_B_single_buf(float* B, float* B_p, int K, int N, int n1,
	            int k_c, int n_c, int n_r, bool pad_n) ;
void pack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n);


void pack_ob_A_multiple_buf(float* A, float* A_p, int M, int K, int m1, int k1, int m2, int m_c, int k_c, int m_r, bool pad) ;
void pack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad);
void pack_ob_B_parallel(float* B, float* B_p, int K, int N, int n1,
            int k_c, int n_c, int n_r, bool pad_n);


void pack_ob_A_parallel(float* A, float* A_p, int M, int K, int m1, 
				int m2, int m_c, int k_c, int m_r, bool pad);


// sparse packing functions
void pack_A_sp_k_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_A_sp_m_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_A_sp_n_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_ob_A_sp(float* A, float* A_p, char* nnz_outer, int* k_inds, char* loc_m, 
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad);
void pack_A_sp(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);



void pack_A_csr_to_sp_k_first(csr_t* csr, int M, int K, int nz, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void csr_to_ob_A_sp(float* vals, int* colind_csr, int* rowptr_csr, int* nnz_tiles, int* num_col_tile,
   char* nnz_outer, int* k_inds, char* loc_m, float* A_p, int M, int m1, int m2, int k1,
   int m_c, int k_c, int m_r, int nz_in, int col_tile_in, int* ret);



// unpacking
void unpack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);
void unpack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);

void unpack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_n_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_rsc(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);
void unpack_C_old(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);



// helper funcs
size_t cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);
size_t cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
size_t cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);


void pack_A_mr_x_kc(float* A, float* A_p, int K, int k_c, int m_r);
void pack_B_nr_x_kc(float* B, float* B_p, int N, int k_c, int n_r);



sp_pack_t* malloc_sp_pack(int M, int K, int nz, blk_dims_t* x, cake_cntx_t* cake_cntx);
void free_sp_pack(sp_pack_t* x);
void sp_pack_to_file(sp_pack_t* sp_pack, char* fname);
void file_to_sp_pack(sp_pack_t* sp_pack, char* fname);

int mat_to_csr_file(float* A, int M, int K, char* fname);
void test_csr_convert(int M, int K, float sparsity);
csr_t* file_to_csr(char* fname);
void csr_to_mat(float* A, int M, int K, int* rowptr, float* vals, int* colind);
void free_csr(csr_t* x);



// kernel helper functions
inline void A_packing_kernel
(       
	int               cdim0,
	int               k0,
	float*      kappa,
	float*      a, int inca0, int lda0,
	float*      p,              int ldp0
) {


#ifdef USE_CAKE_PACK
	pack_A_mr_x_kc(a, p, inca0, k0, cdim0);
#elif USE_BLIS_ARMV8_PACK
	bli_spackm_armv8a_int_8xk(cdim0, k0, kappa, a, inca0, lda0, p, ldp0);
#elif USE_BLIS_HASWELL_PACK
	bli_spackm_haswell_asm_6xk_new(cdim0, k0, kappa, a, inca0, lda0, p, ldp0);
#else
	pack_A_mr_x_kc(a, p, inca0, k0, cdim0);
#endif

}



inline void B_packing_kernel
(       
	int               cdim0,
	int               k0,
	float*      kappa,
	float*      a, int inca0, int lda0,
	float*      p,              int ldp0
) {


#ifdef USE_CAKE_PACK
	pack_B_nr_x_kc(a, p, lda0, k0, cdim0);
#elif USE_BLIS_ARMV8_PACK
	bli_spackm_armv8a_int_12xk(cdim0, k0, kappa, a, inca0, lda0, p, ldp0);
#elif USE_BLIS_HASWELL_PACK
	bli_spackm_haswell_asm_16xk_new(cdim0, k0, kappa, a, inca0, lda0, p, ldp0);
#else
	pack_B_nr_x_kc(a, p, lda0, k0, cdim0);
#endif

}




