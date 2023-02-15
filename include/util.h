#include "common.h"


// Util functions
int run_tests();

bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K);

bool add_checker(float** C_arr, float* C, int M, int N, int p);

void rand_init(float* mat, int r, int c);

void print_array(float* arr, int len);

void print_mat(float* arr, int r, int c);

void print_schedule(enum sched sch);

float rand_gen();

float normalRandom();

bool mat_equals(float* C, float* C_check, int M, int N);
