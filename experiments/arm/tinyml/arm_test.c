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
    int p = 4;
    omp_set_num_threads(p);

    float *A, *B, *C;
    int m, n, k, i, j;
    float alpha, beta;


    int Ms[] = {16,16,32,32,32,64,64,64, 64, 64, 8 ,16 ,32 ,32 ,64 ,64 ,128,128,256,256};
    int Ks[] = {27 ,144,144,288,16 ,288,576,32 , 40, 64, 27 , 8 , 16 , 32 , 32 , 64 , 64 , 128, 128, 256};
    int Ns[] = {1024,1024,256,256,256,64,64,64, 122, 125, 2304,2304,576,576,144,144, 36, 36, 9, 9};

    for(int t = 0; t < 20; t++) {

        m = Ms[t], k = Ks[t], n = Ns[t];


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

        srand(time(NULL));
        rand_init(A, m, k);
        rand_init(B, k, n);

        int ntrials = atoi(argv[1]);

        float ressss;
        float tttmp[18];
        int flushsz=100000;
        diff_t = 0.0;

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

        char fname[50];
        snprintf(fname, sizeof(fname), "results");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "armpl,%d,%d,%d,%d,%d,%f\n", t+1,p,m,k,n, diff_t / ntrials);
        fclose(fp);


        free(A);
        free(B);
        free(C);
    }


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
