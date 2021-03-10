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
    struct timeval start, end;
    double diff_t;
    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

    // mkl_set_num_threads(atoi(argv[1]));
    omp_set_num_threads(atoi(argv[1]));

    float *A, *B, *C;
    int m, n, k, i, j;
    float alpha, beta;

    m = 4000, k = 4000, n = 4000;
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

    gettimeofday (&start, NULL);

    rand_init(A, m, k);
    rand_init(B, k, n);


    // for (i = 0; i < (m*k); i++) {
    //     // A[i] = (double)(i+1);
    //     A[i] = (double)(i);
    // }

    // for (i = 0; i < (k*n); i++) {
    //     // B[i] = (double)(-i-1);
    //     B[i] = (double)(i);
    // }

    // for (i = 0; i < (m*n); i++) {
    //     C[i] = 0.0;
    // }

    gettimeofday (&end, NULL);
    diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    +end.tv_usec) - start.tv_usec) / (1000000.0);
    printf("init time: %f \n", diff_t); 

    gettimeofday (&start, NULL);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    gettimeofday (&end, NULL);
    diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    +end.tv_usec) - start.tv_usec) / (1000000.0);
    printf("GEMM time: %f \n", diff_t); 


    printf ("\n Computations completed.\n\n");


//	for(int i = 0; i < m*n; i++) {
//		printf("%f ", C[i]);
//	}


    printf ("\n Deallocating memory \n\n");
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

