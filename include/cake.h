#pragma once

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include <pthread.h>
#include "blis.h"
 

#define DEBUG 1
#define ARR_PRINT 0
#define CHECK_PRINT 0


extern int m_c;
extern int k_c;
extern int n_c;
extern int m_c1;
extern int k_c1;
extern int n_c1;
extern int m_c1_last_core;
extern int k_c1_last_core;
extern int mr_rem;
extern int nr_rem;
extern int k_rem;
extern int p_l;
extern int m_pad;
extern int k_pad;
extern int n_pad;
extern int Mb;
extern int Kb;
extern int Nb;
extern int M_padded;
extern int N_padded;


enum sched {KMN, MKN, NKM};


typedef struct cake_cntx_t{
	cntx_t* blis_cntx;
	double alpha_n;
	int mr;
	int nr;
	int L2;
	int L3;
	int ncores;
} cake_cntx_t;


typedef struct blk_dims_t{
	int m_c;
	int k_c;
	int n_c;
} blk_dims_t;



struct gemm_input {
	float* A; 
	float* B;
	float* C;
	float alpha;
	float beta;
	int M;
	int N;
	int K; 
	int p;
	cake_cntx_t* cake_cntx;
	bool packedA;
	bool packedB;
};


cake_cntx_t* cake_query_cntx();
cake_cntx_t* cake_query_cntx_torch(int L2, int L3);
/*

Pack 

*/
double pack_A_single_buf_k_first(float* A, float* A_p, int M, int K, int p, cake_cntx_t* cake_cntx);
void pack_B_k_first(float* B, float* B_p, int K, int N, cake_cntx_t* cake_cntx);
void pack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx);

double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, cake_cntx_t* cake_cntx);
void pack_B_m_first(float* B, float* B_p, int K, int N, int p, cake_cntx_t* cake_cntx);
void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx);

// double pack_A_multiple_buf(float* A, float** A_p, int M, int K, int m_c, int k_c, int m_r, int p);
// void pack_C_multiple_buf(float* C, float** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n);


void unpack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, cake_cntx_t* cake_cntx);
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


int cake_sgemm_packed_A_size(int M, int K, int p, cake_cntx_t* cake_cntx);
int cake_sgemm_packed_B_size(int K, int N, int p, cake_cntx_t* cake_cntx);
int cake_sgemm_packed_C_size(int M, int N, int p, cake_cntx_t* cake_cntx);

double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0);

double cake_sgemm_k_first(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0);
double cake_sgemm_m_first(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA = 0, bool packedB = 0, float alpha = 1, float beta = 0);


void* cake_sgemm_launch(void* inputs);


// block sizing and system cache size querying
blk_dims_t* get_block_dims(cake_cntx_t* cake_cntx, int M, int p, enum sched sch);

void init_block_dims(int M, int N, int K, int p, cake_cntx_t* cake_cntx, enum sched sch) ;

int get_cache_size(int level);

int lcm(int n1, int n2);

int get_num_physical_cores();



// Util functions
int run_tests();

bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K);

bool add_checker(float** C_arr, float* C, int M, int N, int p);

void rand_init(float* mat, int r, int c);

void print_array(float* arr, int len);



