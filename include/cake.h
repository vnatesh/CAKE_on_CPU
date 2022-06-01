#pragma once

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include <pthread.h>
 
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef USE_BLIS
#include "blis.h"

#elif USE_CAKE_ARMV8
#include <arm_neon.h>

#elif USE_CAKE_HASWELL
#include <immintrin.h>

#endif



#define DEBUG 1
#define ARR_PRINT 0
#define CHECK_PRINT 0


enum sched {KMN, MKN, NKM, NA};


typedef struct blk_dims_t {
	int m_c;
	int k_c;
	int n_c;
	int m_c1;
	int k_c1;
	int n_c1;
	int m_c1_last_core;
	int k_c1_last_core;
	int mr_rem;
	int nr_rem;
	int k_rem;
	int p_l;
	int m_pad;
	int k_pad;
	int n_pad;
	int Mb;
	int Kb;
	int Nb;
	int M_padded;
	int N_padded;
} blk_dims_t;


typedef struct cake_cntx_t{
	void* blis_cntx;
	double alpha_n;
	double peak_dram_bw;
	double peak_flops;
	int mr;
	int nr;
	int L2;
	int L3;
	int ncores;
} cake_cntx_t;


typedef struct cache_dims_t{
	int m_c;
	int k_c;
	int n_c;
	enum sched sch;
} cache_dims_t;








// sparse matrix handling

typedef struct sp_pack_t {
   int* loc_m; // M dim C writeback location for each nnz value in A
   int* nnz_outer; // number of nnz in every outer prod col vec (with len m_r) of A;
   int* k_inds; // density-based reorder indices of A cols within a mrxkcxnr tile
   int* nnz_outer_blk; // number of nonzeros in each mrxkcxnr outer product blk
   float* A_sp_p; //sparse packed A (only storing nonzeros)
} sp_pack_t;



void pack_A_sp_k_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_A_sp_m_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_A_sp_n_first(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_ob_A_sp(float* A, float* A_p, int* nnz_outer, int* k_inds, int* loc_m, 
   int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad);
void pack_A_sp(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);


void schedule_KMN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_sp(sp_pack_t* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch);


void cake_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k);
// void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
// 							int* nnz_outer, int* loc_m);

void cake_sp_sgemm_haswell_6x16(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* k_inds, int* loc_m);

void cake_sp_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k, 
							int* nnz_outer, int* k_inds, int* loc_m);
void cake_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k);




double cake_sp_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float sparsity = 0, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);


int run_tests_sparse();
void rand_sparse(float* mat, int r, int c, float sparsity);
float rand_gen();
float normalRandom();
void rand_sparse_gaussian(float* mat, int r, int c, float mu, float sigma);





// small matrix handling
bool cake_gemm_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, char* argv[] = NULL);
void cake_sgemm_haswell_6x16_unpacked(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);
void cake_sgemm_haswell_6x16_A_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);
void cake_sgemm_haswell_6x16_B_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);
void cake_sgemm_haswell_6x16_C_packed(float* A, float* B, float* C, int m, int n, int k, int M, int K, int N);



void schedule_KMN_small(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_small(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM_small(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);

void schedule_NKM_small_A_packed(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_small_B_packed(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_small_C_packed(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);

bool cake_gemm_small(float* A, float* A_p, float* B, float* B_p, float* C, float* C_p, 
	int M, int N, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);


// choose cake schedule based on M,N,K values
enum sched set_schedule(enum sched sch, int M, int N, int K);
enum sched derive_schedule(int M, int N, int K, int p, 
					int mc_ret, cake_cntx_t* cake_cntx);

enum sched print_schedule(enum sched sch) ;






// Kernel helper funcs
inline void cake_sgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, cake_cntx_t* cake_cntx);
inline void cake_spgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, int* nnz_outer, int* k_inds, int* loc_m);
inline void cake_sgemm_small_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, int M, int K, int N);




cake_cntx_t* cake_query_cntx();
cake_cntx_t* cake_query_cntx_torch(int L2, int L3);
/*

Pack 

*/

void pack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
void pack_B(float* B, float* B_p, int K, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
double pack_A(float* A, float* A_p, int M, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
void unpack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;


double pack_A_single_buf_k_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_k_first(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_m_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_n_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;
void pack_B_n_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;
void pack_C_single_buf_n_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;

// double pack_A_multiple_buf(float* A, float** A_p, int M, int K, int m_c, int k_c, int m_r, int p);
// void pack_C_multiple_buf(float* C, float** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n);


void unpack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_n_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_rsc(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);
void unpack_C_old(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);


void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, 
				int m2, int m_c, int k_c, int m_r, bool pad);
void pack_ob_B_single_buf(float* B, float* B_p, int K, int N, int n1,
	            int k_c, int n_c, int n_r, bool pad_n) ;
void pack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n);


void pack_ob_A_multiple_buf(float* A, float* A_p, int M, int K, int m1, int k1, int m2, int m_c, int k_c, int m_r, bool pad) ;
void pack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad);


void unpack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);
void unpack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);

int cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);
int cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
int cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);



double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);
void schedule(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch, bool sparse, bool small);
void schedule_KMN(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) ;
void schedule_MKN(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);



// block sizing and system parameter querying
cache_dims_t* get_cache_dims(int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], float sparsity = 0);

void init_block_dims(int M, int N, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], float sparsity = 0);

int get_cache_size(int level);

int lcm(int n1, int n2);

int get_num_physical_cores();



// Util functions
int run_tests();

bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K);

bool add_checker(float** C_arr, float* C, int M, int N, int p);

void rand_init(float* mat, int r, int c);

void print_array(float* arr, int len);

void print_mat(float* arr, int r, int c);
