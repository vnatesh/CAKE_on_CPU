/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.
   Copyright (C) 2014, The University of Texas at Austin
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "blis.h"



static const char delims[] = " \t\n";


void rand_init(float* mat, int r, int c);

int main( int argc, char** argv ) {

    struct timespec start, end;
    double diff_t;

    if(argc < 1) {
        printf("Enter DNN benchmark filename\n");
        exit(1);
    }

    int M, K, N;
    int *Ms, *Ks, *Ns;
    float *A, *B, *C;
    float alpha = 1.0, beta = 0.0;

    inc_t rsa, csa;
    inc_t rsb, csb;
    inc_t rsc, csc;

    // set the number of threads
    int p = 4;
    omp_set_num_threads(p);
    rntm_t rntm;
    bli_rntm_init( &rntm );
    bli_rntm_set_num_threads(p, &rntm );

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

    int iters = 100;

    for(int i = 0; i < mm_cnt; i++) {

        M = Ms[i], K = Ks[i], N = Ns[i];
        printf("M = %d, K = %d, N = %d\n", M,K,N);

        A = (float*) malloc(M * K * sizeof( float ));
        B = (float*) malloc(K * N * sizeof( float ));
        C = (float*) calloc(M * N , sizeof( float ));

        rsc = 1; csc = M;
        rsa = 1; csa = M;
        rsb = 1; csb = K;

        // initialize A and B
        srand(time(NULL));
        rand_init(A, M, K);
        rand_init(B, K, N);


        clock_gettime(CLOCK_REALTIME, &start);

        for(int j = 0; j < iters; j++) {
          bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
           M, N, K, &alpha, A, rsa, csa, B, rsb, csb,
                     &beta, C, rsc, csc );
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
