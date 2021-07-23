#include "cake.h"

 
int main( int argc, char** argv ) {

//	run_tests();
	// exit(1);

	// struct timeval start, end;
	struct timespec start, end;
	double diff_t;

    if(argc < 2) {
        printf("Enter number of Cores\n");
        exit(1);
    }

	int M, K, N, p;

	M = 3000;
	K = 3000;
	N = 3000;

    p = atoi(argv[1]);

	printf("M = %d, K = %d, N = %d\n", M,K,N);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         m, n, k, alpha, A, k, B, n, beta, C, n);

	cake_cntx_t* cake_cntx = cake_query_cntx();

	clock_gettime(CLOCK_REALTIME, &start);

	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sgemm time: %f \n", diff_t); 


	
	free(A);
	free(B);
	free(C);

	return 0;
}


