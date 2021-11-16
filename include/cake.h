#pragma once

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include <pthread.h>
#include "blis.h"
 

#define DEBUG 0
#define ARR_PRINT 0
#define CHECK_PRINT 0


enum sched {KMN, MKN, NKM};


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
	cntx_t* blis_cntx;
	double alpha_n;
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
} cache_dims_t;


cake_cntx_t* cake_query_cntx();
cake_cntx_t* cake_query_cntx_torch(int L2, int L3);
/*

Pack 

*/
double pack_A_single_buf_k_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_k_first(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_m_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

// double pack_A_multiple_buf(float* A, float** A_p, int M, int K, int m_c, int k_c, int m_r, int p);
// void pack_C_multiple_buf(float* C, float** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n);


void unpack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
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




int cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
int cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
int cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);



double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0);

double cake_sgemm_k_first(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0);
double cake_sgemm_m_first(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0);


// block sizing and system parameter querying
cache_dims_t* get_cache_dims(cake_cntx_t* cake_cntx, int M, int p, enum sched sch);

void init_block_dims(int M, int N, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);

int get_cache_size(int level);

int lcm(int n1, int n2);

int get_num_physical_cores();



// Util functions
int run_tests();

bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K);

bool add_checker(float** C_arr, float* C, int M, int N, int p);

void rand_init(float* mat, int r, int c);

void print_array(float* arr, int len);



