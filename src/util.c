#include "cake.h"

void cake_dgemm_checker(double* A, double* B, double* C, int N, int M, int K) {

	double* C_check = calloc(M * N , sizeof( double ));

	#pragma omp parallel for
	for(int m = 0; m < M; m++) {
		for(int n = 0; n < N; n++) {
			C_check[m*N + n] = 0.0;
			// C_check[m*N + n] = C[m*N + n];
			for(int k = 0; k < K; k++) {
				C_check[m*N + n] += A[m*K + k] * B[k*N + n];
			}
			// printf("%f ", C_check[m*N + n]);
		}
		// printf("\n");
	}
	// printf("\n\n\n\n\n");
	// exit(1);

    int CORRECT = 1;
    int cnt = 0;
    int ind1 = 0;
    double eps = 1e-11; // machine precision level

	for(int m = 0; m < M; m++) {
	    for(int n = 0; n < N; n++) {
	        // if(C_check[m1*N + n1] != C[ind1]) {
	        if(fabs(C_check[ind1] - C[ind1]) > eps) {
	            cnt++;
	            CORRECT = 0;
	        }

        printf("%f\t%f\n", C_check[ind1], C[ind1]);

        ind1++; 
      }
    }

    printf("\n\n");

	if(CORRECT) {
		printf("CORRECT!\n");
	} else {
		printf("WRONG!\n");
		printf("%d\n", cnt);
	}

	free(C_check);

	// for (int n1 = 0; n1 < N; n1++) {
 //      for (int m1 = 0; m1 < M; m1++) {
 //        printf("%f ", C_check[m1*N + n1]);
 //      }
 //    }
	// printf("\n\n\n\n");
}



// randomized double precision matrices in range [-1,1]
void rand_init(double* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (double) rand() / RAND_MAX*2.0 - 1.0;
	}	
}



void print_array(double* arr, int len) {

	for(int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n\n");

}
