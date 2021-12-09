#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h> 
#include <string.h>

#include <omp.h>
#include "armpl.h"


// Compile ARMPL test 

// gcc test.c -I/opt/arm/armpl_20.3_gcc-7.1/include -L{ARMPL_DIR} -lm 

// gcc -m64 -I${MKLROOT}/include mkl_sgemm_test.c 
// -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core
//  -lmkl_gnu_thread -lpthread -lm -ldl -o mkl_sgemm_test

void rand_init(float* mat, int r, int c);




static const char delims[] = " \t\n";

int main( int argc, char** argv ) {

    struct timespec start, end;
    double diff_t;

    if(argc < 1) {
        printf("Enter DNN benchmark filename\n");
        exit(1);
    }

    int M, K, N;
    int *Ms, *Ks, *Ns;
    int A_sz, B_sz, C_sz;   
    float *A, *B, *A_p, *B_p, *C;
    float alpha = 1.0, beta = 0.0;

    int p = 4;
    omp_set_num_threads(p);

    FILE* fp = fopen(argv[1],"r");
    if(!fp) {
        printf("Error: Could not open file\n");
        exit(1);
    }

    // first line contains # of MMs
    char line[15];
    int i = 0;
    fgets(line, 15, fp);
    int mm_cnt = atoi(line);
    Ms = (int*) malloc(sizeof(int) * mm_cnt); 
    Ks = (int*) malloc(sizeof(int) * mm_cnt); 
    Ns = (int*) malloc(sizeof(int) * mm_cnt); 
    
    while(fgets(line, 20, fp)) {
        Ms[i] = atoi(strtok(line, delims));
        Ks[i] = atoi(strtok(NULL, delims));
        Ns[i] = atoi(strtok(NULL, delims));
        i++;
    }

    fclose(fp);

    int iters = 10000;

    for(int i = 0; i < mm_cnt; i++) {

        M = Ms[i], K = Ks[i], N = Ns[i];
        printf("M = %d, K = %d, N = %d\n", M,K,N);

        A = (float*) malloc(M * K * sizeof( float ));
        B = (float*) malloc(K * N * sizeof( float ));
        C = (float*) calloc(M * N , sizeof( float ));

        // initialize A and B
        srand(time(NULL));
        rand_init(A, M, K);
        rand_init(B, K, N);

        clock_gettime(CLOCK_REALTIME, &start);

        for(int j = 0; j < iters; j++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
        }

        clock_gettime(CLOCK_REALTIME, &end);
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
        printf("sgemm time: %f \n", diff_t / iters); 

        free(A);
        free(B);
        free(C);
    }



    // char fname[50];
    // snprintf(fname, sizeof(fname), "results_sq");
    // FILE *fp;
    // fp = fopen(fname, "a");
    // fprintf(fp, "cake,%d,%d,%f\n",p,M,diff_t);
    // fclose(fp);


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

