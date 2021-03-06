#include "cake.h"

int run_tests() {

	// float *A, *B, *C;
	int M, K, N, m, k, n, max_threads,p;
	max_threads = omp_get_max_threads() / 2;
	int num_tests = 6;
	int Ms[num_tests] = {1,10,96,111,960,2111};
	int Ks[num_tests] = {1,10,96,111,960,2111};
	int Ns[num_tests] = {1,10,96,111,960,2111};
	int cnt = 0;

	for(p = 2; p <= max_threads; p++)  {
		for(m = 0; m < num_tests; m++) {
			for(k = 0; k < num_tests; k++) {
				for(n = 0; n < num_tests; n++) {
					M = Ms[m];
					K = Ks[k];
					N = Ns[n];

					float* A = (float*) malloc(M * K * sizeof( float ));
					float* B = (float*) malloc(K * N * sizeof( float ));
					float* C = (float*) calloc(M * N , sizeof( float ));
				    srand(time(NULL));

					rand_init(A, M, K);
					rand_init(B, K, N);

					cake_dgemm(A, B, C, M, N, K, p, NULL);
					if(cake_dgemm_checker(A, B, C, N, M, K)) {
						printf("TESTS FAILED on p=%d m=%d k=%d n=%d\n",p,m,k,n);
						cnt++;
					}

					free(A);
					free(B);
					free(C);
				}
			}
		}
	}

	if(cnt) {
		printf("FAILED\n");
	} else {
		printf("ALL TESTS PASSED!\n");
	}

	return 0;
}


bool cake_dgemm_checker(float* A, float* B, float* C, int N, int M, int K) {

	float* C_check = (float*) calloc(M * N , sizeof( float ));

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
    float eps = 1e-3; // machine precision level

	for(int m = 0; m < M; m++) {
	    for(int n = 0; n < N; n++) {
	        // if(C_check[m1*N + n1] != C[ind1]) {
	        if(fabs(C_check[ind1] - C[ind1]) > eps) {
	            cnt++;
	            CORRECT = 0;
	        }

        if(CHECK_PRINT) printf("%f\t%f\n", C_check[ind1], C[ind1]);

        ind1++; 
      }
    }

    //printf("\n\n");

	if(CORRECT) {
		printf("CORRECT!\n");
		return 0;
	} else {
		if(CHECK_PRINT) {
			printf("WRONG!\n");
			printf("%d\n", cnt);
		}	
		return 1;
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
void rand_init(float* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
	}	
}



void print_array(float* arr, int len) {

	for(int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n\n");

}
