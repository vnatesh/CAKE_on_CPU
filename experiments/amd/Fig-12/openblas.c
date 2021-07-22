#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include <omp.h>
#include <time.h>
#include <sys/time.h>


void rand_init(float* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
	}	
}

int main(int argc, char *argv[]) {
  
  struct timespec start, end;
  double diff_t;
   
  openblas_set_num_threads(atoi(argv[1]));
  float *A, *B, *C;
  int m, n, k, i, j;
  float alpha, beta;

  // m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
  m = 23040, k = 23040, n = 23040;
  // printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
  //        " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
  alpha = 1.0; beta = 0.0;

  //printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
  //        " performance \n\n");
  A = (float *) malloc( m*k*sizeof( float ));
  B = (float *) malloc( k*n*sizeof( float ));
  C = (float *) calloc( m*n,sizeof( float ));
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

  //start = clock();
  //gettimeofday (&start, NULL);
  clock_gettime(CLOCK_REALTIME, &start);

  //#pragma omp parallel
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            m, n, k, alpha, A, k, B, n, beta, C, n);
  
  clock_gettime(CLOCK_REALTIME, &end);
  long seconds = end.tv_sec - start.tv_sec;
  long nanoseconds = end.tv_nsec - start.tv_nsec;
  diff_t = seconds + nanoseconds*1e-9;

  //gettimeofday (&end, NULL);
  //diff_t = (((end.tv_sec - start.tv_sec)*1000000L
  //+end.tv_usec) - start.tv_usec) / (1000000.0);
  printf("GEMM time: %f \n", diff_t);
  //end = clock();
  //cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC*32);
    //printf ("\n Computations completed. Took %f seconds\n\n", cpu_time_used);


  //printf ("\n Deallocating memory \n\n");
  free(A);
  free(B);
  free(C);

  return 0;
}
