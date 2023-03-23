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


void rand_init(float* mat, int r, int c) {
    // int MAX = 65536;
    for(int i = 0; i < r*c; i++) {
        // mat[i] = (double) i;
        // mat[i] = 1.0;
        // mat[i] =  (double) (i%MAX);
        mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
    }   
}


int main(int argc, char* argv[])  {
    // struct timeval start, end;
    struct timespec start, end;
    double diff_t;
    // printf("max threads %d\n\n", mkl_get_max_threads());
    if(argc < 2) {
        printf("Enter number of threads and dim size\n");
        exit(1);
    }

    int p = atoi(argv[4]);
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


    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);




    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );

    srand(time(NULL));
    rand_init(A, m, k);
    rand_init(B, k, n);

    int ntrials = atoi(argv[5]);
    float ressss;
    float tttmp[18];
    int flushsz=10000000;

    for(int i = 0; i < ntrials; i++) {


        float *dirty = (float *)malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


        clock_gettime(CLOCK_REALTIME, &start);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
        clock_gettime(CLOCK_REALTIME, &end);

        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t += seconds + nanoseconds*1e-9;

        free(dirty);
    }


    printf("GEMM time: %f \n", diff_t / ntrials); 

    // char fname[50];
    // snprintf(fname, sizeof(fname), "results_sq");
    // FILE *fp;
    // fp = fopen(fname, "a");
    // fprintf(fp, "mkl,%d,%d,%f\n",p,m,diff_t);
    // fclose(fp);


    // printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    // printf (" Example completed. \n\n");
    return 0;
}



