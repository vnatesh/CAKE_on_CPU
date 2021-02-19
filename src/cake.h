#include <stdio.h>
#include <sys/time.h> 
#include <time.h> 
#include <omp.h>
#include "blis.h"
 

#define DEBUG 0

void pack_B(double* B, double* B_p, int K, int N, int k_c, int n_c, int n_r, int alpha_n, int m_c);


void pack_A(double* A, double** A_p, int M, int K, int m_c, int k_c, int m_r, int p);


void pack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int m_r, int n_r, int p, int alpha_n);


void unpack_C_rsc(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);


void unpack_C(double* C, double** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);


void set_ob_A(double* A, double* A_p, int M, int K, int m1, int k1, 
				int m2, int m_c, int k_c, int m_r, bool pad);


void set_ob_C(double* C, double* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad);


void cake_dgemm(double* A, double* B, double* C, int M, int N, int K, int p);


int get_block_dim(int m_r, int n_r, double alpha_n);


int get_cache_size(char* level);

int lcm(int n1, int n2);

void cake_dgemm_checker(double* A, double* B, double* C, int N, int M, int K);

void rand_init(double* mat, int r, int c);

void print_array(double* arr, int len);

