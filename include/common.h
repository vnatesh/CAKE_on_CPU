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


#define DEBUG 0
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
	enum sched sch;
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



