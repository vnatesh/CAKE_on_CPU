#pragma once

#include "kernels.h"
#include "util.h"
#include "tiling.h"
#include "packing.h"



// Dense MM scheduling
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



void schedule_KMN_online(float* A, float* B, float* C, float** A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_C_unpacked(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_2d(float* A, float* B, float* C, float* A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_2d_small(float* A, float* B, float* C, float* A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);


double cake_sgemm_online(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);
double cake_sgemm_test(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);
double cake_sgemm_2d(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);
double cake_sgemm_2d_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);


// sparse MM scheduling
double cake_sp_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, bool packedA = 0, 
	sp_pack_t* sp_pack = NULL, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);
void schedule_KMN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM_sp(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_sp(sp_pack_t* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch);





double cake_sp_sgemm_testing(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density = 0, char* argv[] = NULL, char* fname = NULL, bool packedA = 0, 
	bool packedB = 0, float alpha = 1, float beta = 0, enum sched sch = NA);
void schedule_KMN_sp_test(sp_pack_t* sp_pack, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);




// small matrix handling
bool cake_gemm_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, char* argv[] = NULL);



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




