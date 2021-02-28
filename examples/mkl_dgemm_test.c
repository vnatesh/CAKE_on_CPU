#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h> 
#include "mkl.h"



void rand_init(double* mat, int r, int c);


int main(int argc, char* argv[])  {
    struct timeval start, end;
    double diff_t;
    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

    mkl_set_num_threads(atoi(argv[1]));

    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    m = 23095, k = 23095, n = 23095;
    // m = 30720, k = 30720, n = 30720;  
    //  m = 25921, k = 25921, n = 25921;      
    // m = 23040, k = 23040, n = 23040;
    alpha = 1.0; beta = 0.0;

    A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );

    printf("M = %d, K = %d, N = %d\n", m, k, n);

    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    gettimeofday (&end, NULL);
    diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    +end.tv_usec) - start.tv_usec) / (1000000.0);
    printf("GEMM time: %f \n", diff_t); 



    printf ("\n Computations completed.\n\n");

    // printf (" Top left corner of matrix A: \n");
    // for (i=0; i<min(m,6); i++) {
    //   for (j=0; j<min(k,6); j++) {
    //     printf ("%12.0f", A[j+i*k]);
    //   }
    //   printf ("\n");
    // }

    // printf ("\n Top left corner of matrix B: \n");
    // for (i=0; i<min(k,6); i++) {
    //   for (j=0; j<min(n,6); j++) {
    //     printf ("%12.0f", B[j+i*n]);
    //   }
    //   printf ("\n");
    // }
   
    // printf ("\n Top left corner of matrix C: \n");
    // for (i=0; i<min(m,6); i++) {
    //   for (j=0; j<min(n,6); j++) {
    //     printf ("%12.5G", C[j+i*n]);
    //   }
    //   printf ("\n");
    // }



    // double* C_check = malloc(m * n * sizeof( double ));
    // for (int n1 = 0; n1 < n; n1++) {
    //   for (int m1 = 0; m1 < m; m1++) {
    //     C_check[m1*n + n1] = 0.0;
    //     for (int k1 = 0; k1 < k; k1++) {
    //       C_check[m1*n + n1] += A[m1*k + k1] * B[k1*n + n1];
    //     }
    //   }
    // }


    // int CORRECT = 1;
    // int cnt = 0;
    // int* wrong = (int*) malloc(m*n*sizeof(int));
    // int* corrects = (int*) malloc(m*n*sizeof(int));

    // for (int n1 = 0; n1 < n; n1++) {
    //   for (int m1 = 0; m1 < m; m1++) {
    //     // printf("%f ", C_check[m*N + n]);
    //     if(C_check[m1*n + n1] != C[m1*n + n1]) {
    //         wrong[cnt] = m1*n + n1;
    //         cnt++;
    //         CORRECT = 0;
    //     } 
    //   }
    // }

    // printf("\n\n");

    // // printf("wrongs:");
    // // for(int i = 0; i < 100; i++) {
    // //     printf(" %d ", wrong[i]);
    // // }
    // // printf("\n\n");

    // printf("wrongs:");
    // for(int i = 0; i < 100; i++) {
    //     printf(" %f ", C_check[wrong[i]]);
    // }
    // printf("\n\n");

    // printf("corrects:");
    // for(int i = 0; i < 100; i++) {
    //     printf(" %f ", C[wrong[i]]);
    // }
    // printf("\n\n");


    // if(CORRECT) {
    //   printf("CORRECT!\n");
    // } else {
    //   printf("WRONG!\n");
    //     printf("%d\n", cnt);
    // }



    printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf (" Example completed. \n\n");
    return 0;
}



void rand_init(double* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (double) rand() / RAND_MAX*2.0 - 1.0;
    }   
}

