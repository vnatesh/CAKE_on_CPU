#include "cake.h"




int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p;
	struct timespec start, end;
	double diff_t;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = atoi(argv[4]);

	printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));

    float sparsity = 0.90;
	rand_sparse(A, M, K, sparsity);
	// rand_sparse_gaussian(A, M, K, 0, 1);
	// rand_init(A, M, K);
	// print_array(A, M*K);
	// exit(1);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	
//	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
	
	clock_gettime(CLOCK_REALTIME, &start);

	cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, 1-sparsity);

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sp_sgemm time: %f \n", diff_t); 


	clock_gettime(CLOCK_REALTIME, &start);

	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

    clock_gettime(CLOCK_REALTIME, &end);
     seconds = end.tv_sec - start.tv_sec;
     nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sgemm time: %f \n", diff_t); 


	cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);
	free(cake_cntx);
	
	return 0;
}




