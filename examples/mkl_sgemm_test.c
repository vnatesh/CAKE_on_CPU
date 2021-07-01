#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h> 
#include "mkl.h"

// Compile MKL test file using the intel advisor below:
// https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html

// gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c 
// -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core
//  -lmkl_gnu_thread -lpthread -lm -ldl -o mkl_sgemm_test

void rand_init(float* mat, int r, int c);


int main(int argc, char* argv[])  {
    // struct timeval start, end;
    struct timespec start, end;
    double diff_t;
    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads and dim size\n");
        exit(1);
    }

     //int p = atoi(argv[1]);
    // mkl_set_num_threads(atoi(argv[1]));
    int p = 10;
    mkl_set_num_threads(p);

    float *A, *B, *C;
    int m, n, k, i, j;
    float alpha, beta;

    m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
    //m = atoi(argv[2]);
    //k = m;
    //n = m;

    // m = 3000, k = 3000, n = 3000;  
    //  m = 25921, k = 25921, n = 25921;      
    // m = 23040, k = 23040, n = 23040;
    alpha = 1.0; beta = 0.0;

    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );

    printf("M = %d, K = %d, N = %d\n", m, k, n);

    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    // gettimeofday (&start, NULL);
    srand(time(NULL));
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

    // gettimeofday (&end, NULL);
    // diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    // +end.tv_usec) - start.tv_usec) / (1000000.0);
    // printf("init time: %f \n", diff_t); 



    // gettimeofday (&start, NULL);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             m, n, k, alpha, A, k, B, n, beta, C, n);
    // gettimeofday (&end, NULL);
    // diff_t = (((end.tv_sec - start.tv_sec)*1000000L
    // +end.tv_usec) - start.tv_usec) / (1000000.0);
    // printf("GEMM time: %f \n", diff_t); 

    clock_gettime(CLOCK_REALTIME, &start);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    clock_gettime(CLOCK_REALTIME, &end);

    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("GEMM time: %f \n", diff_t); 

    // char fname[50];
    // snprintf(fname, sizeof(fname), "results_sq");
    // FILE *fp;
    // fp = fopen(fname, "a");
    // fprintf(fp, "mkl,%d,%d,%f\n",p,m,diff_t);
    // fclose(fp);


    // printf ("\n Computations completed.\n\n");

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



    // printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    // printf (" Example completed. \n\n");
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

