#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h> 

#include <omp.h>
#include "armpl.h"


// Compile ARMPL test 

// gcc test.c -I/opt/arm/armpl_20.3_gcc-7.1/include -L{ARMPL_DIR} -lm 

// gcc -m64 -I${MKLROOT}/include mkl_sgemm_test.c 
// -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core
//  -lmkl_gnu_thread -lpthread -lm -ldl -o mkl_sgemm_test

void rand_init(float* mat, int r, int c);


int main(int argc, char* argv[])  {
    struct timespec start, end;
    double diff_t;
    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

    // mkl_set_num_threads(atoi(argv[1]));
    int p = atoi(argv[4]);
    omp_set_num_threads(p);

    float *A, *B, *C;
    int m, n, k, i, j;
    float alpha, beta;

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);


    alpha = 1.0; beta = 0.0;

    A = (float *) malloc(m * k * sizeof(float));
    B = (float *) malloc(k * n * sizeof(float));
    C = (float *) malloc(m * n * sizeof(float));

    printf("M = %d, K = %d, N = %d\n", m, k, n);

    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      free(A);
      free(B);
      free(C);
      return 1;
    }


    rand_init(A, m, k);
    rand_init(B, k, n);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);

    clock_gettime(CLOCK_REALTIME, &start);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);


    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("sgemm time: %f \n", diff_t); 


    char fname[50];
    snprintf(fname, sizeof(fname), "results_sq");
    FILE *fp;
    fp = fopen(fname, "a");
    fprintf(fp, "armpl,%d,%d,%f\n",p,m,diff_t);
    fclose(fp);


    free(A);
    free(B);
    free(C);

    printf (" Example completed. \n\n");
    return 0;
}



void rand_init(float* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
    }   
}

