#include <stdio.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include "blis.h"
 

#define DEBUG 0
#define ARR_PRINT 0
#define CHECK_PRINT 0


typedef struct cake_cntx_t{
	cntx_t* blis_cntx;
	double alpha;
	int mr;
	int nr;
	int L2;
	int L3;
} cake_cntx_t;


cake_cntx_t* cake_query_cntx();
cake_cntx_t* cake_query_cntx_torch(int L2, int L3);
/*

Pack 

*/
void pack_B(float* B, float* B_p, int K, int N, int k_c, int n_c, int n_r, int alpha_n, int m_c);


void pack_A(float* A, float** A_p, int M, int K, int m_c, int k_c, int m_r, int p);


void pack_C(float* C, float** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n);


void unpack_C_rsc(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);


void unpack_C(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);


void pack_ob_A(float* A, float* A_p, int M, int K, int m1, int k1, 
				int m2, int m_c, int k_c, int m_r, bool pad);


void pack_ob_C(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad);



void unpack_ob_C(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);



void cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, cake_cntx_t* cake_cntx);


int get_block_dim(cake_cntx_t* cake_cntx, int M, int p);


int get_cache_size(int level);

int lcm(int n1, int n2);

int run_tests();

bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K);

void rand_init(float* mat, int r, int c);

void print_array(float* arr, int len);

