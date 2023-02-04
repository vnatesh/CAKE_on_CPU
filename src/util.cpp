#include "cake.h"

int run_tests() {

	// float *A, *B, *C;
	int M, K, N, m, k, n, max_threads,p;
	float *A, *B, *C;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	max_threads = cake_cntx->ncores;
	int num_tests = 6;
	int Ms[6] = {1,10,96,111,960,2111};
	int Ks[6] = {1,10,96,111,960,2111};
	int Ns[6] = {1,10,96,111,960,2111};
	int cnt = 0;


	for(p = 2; p <= max_threads; p++)  {
		for(m = 0; m < num_tests; m++) {
			for(k = 0; k < num_tests; k++) {
				for(n = 0; n < num_tests; n++) {
					M = Ms[m];
					K = Ks[k];
					N = Ns[n];

					A = (float*) malloc(M * K * sizeof( float ));
					B = (float*) malloc(K * N * sizeof( float ));
					C = (float*) calloc(M * N , sizeof( float ));
				    srand(time(NULL));

					rand_init(A, M, K);
					rand_init(B, K, N);

					cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0,0,1,0, MKN);
					if(cake_sgemm_checker(A, B, C, N, M, K)) {
						printf("TESTS FAILED on M-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
						cnt++;
					}

					free(A);
					free(B);
					free(C);


					A = (float*) malloc(M * K * sizeof( float ));
					B = (float*) malloc(K * N * sizeof( float ));
					C = (float*) calloc(M * N , sizeof( float ));
				    srand(time(NULL));

					rand_init(A, M, K);
					rand_init(B, K, N);

					cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0,0,1,0,KMN);
					if(cake_sgemm_checker(A, B, C, N, M, K)) {
						printf("TESTS FAILED on K-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
						cnt++;
					}

					free(A);
					free(B);
					free(C);


					A = (float*) malloc(M * K * sizeof( float ));
					B = (float*) malloc(K * N * sizeof( float ));
					C = (float*) calloc(M * N , sizeof( float ));
				    srand(time(NULL));

					rand_init(A, M, K);
					rand_init(B, K, N);

					cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0,0,1,0, NKM);
					if(cake_sgemm_checker(A, B, C, N, M, K)) {
						printf("TESTS FAILED on N-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
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



int run_tests_sparse() {

	// float *A, *B, *C;
	int M, K, N, m, k, n, max_threads,p;
	float *A, *B, *C;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	max_threads = cake_cntx->ncores;
	int num_tests = 6;
	int Ms[6] = {1,10,96,111,960,2111};
	int Ks[6] = {1,10,96,111,960,2111};
	int Ns[6] = {1,10,96,111,960,2111};
	int cnt = 0;

	printf("starting spMM tests\n");

	for(p = 2; p <= max_threads; p++)  {
		for(m = 0; m < num_tests; m++) {
			for(k = 0; k < num_tests; k++) {
				for(n = 0; n < num_tests; n++) {
					
					M = Ms[m];
					K = Ks[k];
					N = Ns[n];

					A = (float*) malloc(M * K * sizeof( float ));
					B = (float*) malloc(K * N * sizeof( float ));
					C = (float*) calloc(M * N , sizeof( float ));
				    srand(time(NULL));

				    rand_sparse(A, M, K, 0.5);
					rand_init(B, K, N);

					cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, 0.5);
					if(cake_sgemm_checker(A, B, C, N, M, K)) {
						printf("TESTS FAILED on M-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
						cnt++;
					}

					free(A);
					free(B);
					free(C);


					// A = (float*) malloc(M * K * sizeof( float ));
					// B = (float*) malloc(K * N * sizeof( float ));
					// C = (float*) calloc(M * N , sizeof( float ));
				 //    srand(time(NULL));

				 //    rand_sparse(A, M, K, 0.5);
					// rand_init(B, K, N);

					// cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, 0,0,1,0,KMN);
					// if(cake_sgemm_checker(A, B, C, N, M, K)) {
					// 	printf("TESTS FAILED on K-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
					// 	cnt++;
					// }

					// free(A);
					// free(B);
					// free(C);


					// A = (float*) malloc(M * K * sizeof( float ));
					// B = (float*) malloc(K * N * sizeof( float ));
					// C = (float*) calloc(M * N , sizeof( float ));
				 //    srand(time(NULL));

				 //    rand_sparse(A, M, K, 0.5);
					// rand_init(B, K, N);

					// cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, 0,0,1,0, NKM);
					// if(cake_sgemm_checker(A, B, C, N, M, K)) {
					// 	printf("TESTS FAILED on N-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
					// 	cnt++;
					// }

					// free(A);
					// free(B);
					// free(C);
				}
			}
		}
	}

	if(cnt) {
		printf("FAILED\n");
	} else {
		printf("ALL SPARSE MM TESTS PASSED!\n");
	}

	return 0;
}




int run_tests_sparse_test() {

	printf("sparse testing!!!!!\n");
	sleep(7);
	// float *A, *B, *C;
	int M, K, N, m, k, n, max_threads,p;
	float *A, *B, *C;
	cake_cntx_t* cake_cntx = cake_query_cntx();
	max_threads = cake_cntx->ncores;
	int num_tests = 6;
	int Ms[6] = {1,10,96,111,960,2111};
	int Ks[6] = {1,10,96,111,960,2111};
	int Ns[6] = {1,10,96,111,960,2111};
	int cnt = 0;

	printf("starting spMM tests\n");

	for(p = 9; p <= max_threads; p++)  {
		for(m = 0; m < num_tests; m++) {
			for(k = 0; k < num_tests; k++) {
				for(n = 0; n < num_tests; n++) {
					
					M = Ms[m];
					K = Ks[k];
					N = Ns[n];

					A = (float*) malloc(M * K * sizeof( float ));
					B = (float*) malloc(K * N * sizeof( float ));
					C = (float*) calloc(M * N , sizeof( float ));
				    srand(time(NULL));

				    rand_sparse(A, M, K, 0.5);
					rand_init(B, K, N);


					char fname[50];
					snprintf(fname, sizeof(fname), "convert_test");
					int nz = mat_to_csr_file(A, M, K, fname);
					
					csr_t* csr = file_to_csr(fname);
					float density = ((float) csr->rowptr[M]) / ((float) (((float) M) * ((float) K)));
					blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
					cake_cntx_t* cake_cntx = cake_query_cntx();
					init_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL, density);
				    float* A_p = (float*) calloc(nz, sizeof(float));
					sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
					pack_A_csr_to_sp_k_first(csr, A_p, M, K, nz, p, sp_pack, x, cake_cntx);
					cake_sp_sgemm_testing(fname, B, C, M, N, K, p, cake_cntx, density, NULL, sp_pack, 1, 0, 1, 0, KMN);
					
					free_csr(csr);
					free_sp_pack(sp_pack);

					if(cake_sgemm_checker(A, B, C, N, M, K)) {
						printf("TESTS FAILED on M-first p=%d M=%d K=%d N=%d\n",p,M,K,N);
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
		printf("ALL SPARSE MM TESTS PASSED!\n");
	}

	return 0;
}



void free_csr(csr_t* x) {
	free(x->rowptr);
	free(x->colind);
	free(x->vals);
	free(x);
}

void free_sp_pack(sp_pack_t* x) {
	free(x->loc_m);
	free(x->nnz_outer);
	free(x->k_inds);
	free(x->A_sp_p);
	free(x->nnz_tiles);
	free(x->num_col_tile);
}


// read in CSR matrix from file
csr_t* file_to_csr(char* fname) {

	int M, K, nz;

	FILE *fptr;
	char *line = NULL;
	size_t len = 0;
	ssize_t nread;

	fptr = fopen(fname, "r");
	if (fptr == NULL) {
	   perror("fopen");
	   exit(EXIT_FAILURE);
	}

	nread = getline(&line, &len, fptr);
	M = atoi(strtok(line," "));
	K = atoi(strtok(NULL, " "));
	nz = atoi(strtok(NULL, " "));
	float* vals = (float*) malloc(nz * sizeof(float));
	int* rowptr = (int*) malloc((M + 1) * sizeof(int));
	int* colind = (int*) malloc(nz * sizeof(int));

	// printf("M = %d K = %d nz = %d, cores = %d, file = %s\n", M, K, nz, p, argv[9]);

	nread = getline(&line, &len, fptr);

	char* tok;
	tok = strtok(line," ");
	rowptr[0] = atoi(tok);

	for(int i = 1; i < M+1; i++) {
		tok = strtok(NULL, " ");
		rowptr[i] = atoi(tok);
	}

	for(int i = 0; i < nz; i++) {
		tok = strtok(NULL, " ");
		colind[i] = atoi(tok);
	}

	for(int i = 0; i < nz; i++) {
		tok = strtok(NULL, " ");
		vals[i] = atof(tok);
	}

   	free(line);
   	fclose(fptr);

	csr_t* csr_ret = (csr_t*) malloc(sizeof(csr_t));
	csr_ret->rowptr = rowptr;
	csr_ret->colind = colind;
	csr_ret->vals = vals;

   	return csr_ret;
}



void csr_to_mat(float* A, int M, int K, int* rowptr, float* vals, int* colind) {

	int ks, ind = 0;

	for(int i = 0; i < M; i++) {
		ks = rowptr[i+1] - rowptr[i];
		for(int j = 0; j < ks; j++) {
			A[i*K + colind[ind]] = vals[ind];
			ind++;
		}
	}
}


void test_csr_convert(int M, int K, float sparsity) {

	char fname[50];
	snprintf(fname, sizeof(fname), "convert_test");
	float* A = (float*) malloc(M * K * sizeof( float ));
	float* A_check = (float*) malloc(M * K * sizeof(float));

	// initialize A and B
    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sparsity) / 100.0);

	int nz = mat_to_csr_file(A, M, K, fname);
	csr_t* csr = file_to_csr(fname);
	csr_to_mat(A_check, M, K, csr->rowptr, csr->vals, csr->colind);
	mat_equals(A, A_check, M, K);
	free(A); free(A_check);

	// for(int i = 0; i < M+1; i++) {
	// 	printf("%d ", csr->rowptr[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < nz; i++) {
	// 	printf("%d ", csr->colind[i]);
	// }
	// printf("\n");


	// for(int i = 0; i < nz; i++) {
	// 	printf("%f ", csr->vals[i]);
	// }
	// printf("\n");
}



void mat_equals(float* C, float* C_check, int M, int N) {

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
	} else {
		printf("WRONG!\n");
		printf("%d\n", cnt);
	}
}





int mat_to_csr_file(float* A, int M, int K, char* fname) {

	float* vals = (float*) malloc(M * K * sizeof(float));
	int* colind = (int*) malloc(M * K * sizeof(int));
	int* rowptr = (int*) malloc((M+1) * sizeof(int));
	rowptr[0] = 0;

	FILE *fp = fopen(fname, "w");
	int row_cnt, nz = 0;

	for(int i = 0; i < M; i++) {
		row_cnt = 0;
		for(int j = 0; j < K; j++) {
			float tmp = A[i*K + j];
			if(tmp != 0) {
				vals[nz] = tmp;
				colind[nz] = j;
				nz++;
			}
		}

		rowptr[i+1] = nz;
	}

	fprintf(fp, "%d %d %d\n", M, K, nz);
	for(int i = 0; i < (M+1); i++) {
		fprintf(fp, "%d ", rowptr[i]);
	}
	for(int i = 0; i < nz; i++) {
		fprintf(fp, "%d ", colind[i]);
	}
	for(int i = 0; i < nz; i++) {
		fprintf(fp, "%f ", vals[i]);
	}

	fclose(fp);
	free(rowptr); free(vals); free(colind);
	return nz;
}


bool cake_sgemm_checker(float* A, float* B, float* C, int N, int M, int K) {

	float* C_check = (float*) calloc(M * N , sizeof( float ));

	for(int k = 0; k < K; k++) {
		#pragma omp parallel for
		for(int m = 0; m < M; m++) {
			for(int n = 0; n < N; n++) {
			// C_check[m*N + n] = 0.0;
			// C_check[m*N + n] = C[m*N + n];
				C_check[m*N + n] += A[m*K + k] * B[k*N + n];
			}
			// printf("%f ", C_check[m*N + n]);
		}
		// printf("\n");
	}
	// printf("\n\n\n\n\n");
	// exit(1);

	mat_equals(C, C_check, M, N);
	free(C_check);

	// for (int n1 = 0; n1 < N; n1++) {
 //      for (int m1 = 0; m1 < M; m1++) {
 //        printf("%f ", C_check[m1*N + n1]);
 //      }
 //    }
	// printf("\n\n\n\n");
}




bool add_checker(float** C_arr, float* C, int M, int N, int p) {

	float* C_check = (float*) calloc(M * N , sizeof( float ));

	for(int c = 0; c < p; c++) {
		for(int i = 0; i < M*N; i++) {
			C_check[i] +=  C_arr[c][i];
		}
	}

    int CORRECT = 1;
    int cnt = 0;
    int ind1 = 0;
    float eps = 1e-3; // machine precision level

    for(int i = 0; i < M*N; i++) {
        // if(C_check[m1*N + n1] != C[ind1]) {
        if(fabs(C_check[ind1] - C[ind1]) > eps) {
            cnt++;
            CORRECT = 0;
        }

        if(CHECK_PRINT) printf("%f\t%f\n", C_check[ind1], C[ind1]);
        ind1++; 
  	}


    //printf("\n\n");

	if(CORRECT) {
		printf("ADD CORRECT!\n");
		free(C_check);
		return 0;
	} else {
		printf("ADD WRONG!\n");
		printf("%d\n", cnt);
		free(C_check);
		return 1;
	}


}



// randomized double precision matrices in range [-1,1]
void rand_init(float* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (float) rand() / ((float) RAND_MAX)*2.0 - 1.0;
	}	
}



// randomized sparse matrix in range [-1,1] 
// with sparsity % of values that are zero
// i.e., threshold pruning
void rand_sparse(float* mat, int r, int c, float sparsity) {

	for(int i = 0; i < r*c; i++) {
		int x = rand();
		if(x <= ((float) RAND_MAX)*sparsity) {
			mat[i] = 0;
		} else {
			mat[i] =  (float) x / ((float) RAND_MAX)*2.0 - 1.0;
		}
	}	
}



// return a uniformly distributed random value in [0,1]
float rand_gen() {
   return ( (float)(rand()) + 1. ) / ( (float)(((float) RAND_MAX)) + 1. );
}


// Box-Muller transform to generate a normally 
// distributed N(0,1) random value
float normalRandom() {
  
   float v1 = rand_gen();
   float v2 = rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}


// generates random matrix in ~ N(0,1)
void rand_gaussian(float* mat, int r, int c) {

	for(int i = 0; i < r*c; i++) {
		mat[i] =  normalRandom();
	}	
}


// randomized sparse Normal(0,1) matrix with sparsity % of values determined by sigma (std dev)
void rand_sparse_gaussian(float* mat, int r, int c, float mu, float sigma) {
	int nnz = 0;
	for(int i = 0; i < r*c; i++) {
		float x = normalRandom()*sigma+mu;
		if(fabs(x) <= 2) { // 2 sigmas i.e. 95% sparse
			mat[i] = 0;
		} else {
			mat[i] =  x;
			nnz++;
		}
	}	
	printf("nnz = %d\n", nnz);
}


void print_array(float* arr, int len) {

	for(int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n\n");

}


void print_mat(float* arr, int r, int c) {

	for(int i = 0; i < r; i++) {
		for(int j = 0; j < c; j++) {
			printf("%f ", arr[i*c + j]);
		}
		printf("\n");
	}
	printf("\n\n");

}


void print_schedule(enum sched sch) {

	switch(sch) {
		case KMN: {
		    printf("\n KMN Cake Schedule\n");
			break;
		}
		case MKN: {
		    printf("\n MKN Cake Schedule\n");
			break;
		}
		case NKM: {
		    printf("\n NKM Cake Schedule\n");
			break;
		}
		default: {
			printf("\n unknown schedule\n");
			exit(1);
		}	
	}
}



void transpose(float* At, float* A, int M, int K) {

	for(int k = 0; k < K; k++) {
		for(int m = 0; m < M; m++) {
			At[m + k*M] = A[m*K + k];
		}
	}
}
