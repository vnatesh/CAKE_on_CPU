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
#include <stdint.h>
#include <stdbool.h>

#ifdef USE_BLIS
#include "blis.h"
#include "bli_x86_asm_macros.h"



// blis packing kernels
void bli_spackm_haswell_asm_6xk_new
     (
       int               cdim0,
       int               k0,
       float*      kappa,
       float*      a, int inca0, int lda0,
       float*      p,              int ldp0
     );


void bli_spackm_haswell_asm_16xk_new
     (
       int               cdim0,
       int               k0,
       float*      kappa,
       float*      a, int inca0, int lda0,
       float*      p,              int ldp0
     );

#elif USE_CAKE_ARMV8
#include <arm_neon.h>



// blis packing kernels
void bli_spackm_armv8a_int_8xk
     (
       int               cdim0,
       int               k0,
       float*      kappa,
       float*      a, int inca0, int lda0,
       float*      p,              int ldp0
     );


void bli_spackm_armv8a_int_12xk
     (
       int               cdim0,
       int               k0,
       float*      kappa,
       float*      a, int inca0, int lda0,
       float*      p,              int ldp0
     );


#elif USE_CAKE_HASWELL
#include <immintrin.h>
#include "bli_x86_asm_macros.h"



// blis packing kernels
void bli_spackm_haswell_asm_6xk_new
     (
       int               cdim0,
       int               k0,
       float*      kappa,
       float*      a, int inca0, int lda0,
       float*      p,              int ldp0
     );


void bli_spackm_haswell_asm_16xk_new
     (
       int               cdim0,
       int               k0,
       float*      kappa,
       float*      a, int inca0, int lda0,
       float*      p,              int ldp0
     );


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
  int n_c1_last_core;
  int k_c1_last_core;
  int mr_rem;
  int nr_rem;
  int k_rem;
  int p_l;
  int pm_l;
  int pn_l;
  int pm;
  int pn;
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
  int m_map;
  int n_map;
  int L1;
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
  char* loc_m; // nnz vals (3-5 bits) // M dim C writeback location for each nnz value in A (3-5 bits)
  char* nnz_outer; // at most nnz vals (3-5 bits) // number of nnz in every outer prod col vec (with len m_r) of A;
  int* k_inds; // at most nnz vals // density-based reorder indices of A cols within a mrxkcxnr tile
  int* nnz_outer_blk; // number of nonzeros in each mrxkcxnr outer product blk
  float* A_sp_p; // nnz vals (32-bit) // sparse packed A (only storing nonzeros)
  int* nnz_tiles; // (M*K)/(mr*kc) vals // cum-sum number of nnz vals in each mr x kc tile of A
  int* num_col_tile; // (M*K)/(mr*kc) vals // cum-sum number of outer-product cols in each mr x kc tile that have 1 or more nnz vals 
} sp_pack_t;



typedef struct csr_t {
  float* vals; // nnz 32-bit vals
  int* rowptr; // M+1 32-bit vals
  int* colind; // nnz 32-bit vals
} csr_t;


