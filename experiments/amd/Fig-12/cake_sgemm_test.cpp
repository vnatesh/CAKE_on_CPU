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

	 // M = atoi(argv[1]);
	 // K = atoi(argv[2]);
	 // N = atoi(argv[3]);
	 //M = atoi(argv[2]);
	 //K = M;
	 //N = M;
	// M = 3000;
	// K = 3000;
	// N = 3000;


	// M = 128;
	// K = 3072;
	// N = 768;

	// M = 96;
	// K = 14583;
	// N = 96;
	// m_c = 96;
	// k_c = 96;

	M = 23040;
	K = 23040;
	N = 23040;

	// M = 960;
	// K = 960;
	// N = 960;

    p = atoi(argv[1]);
    // p = 10;

	printf("M = %d, K = %d, N = %d\n", M,K,N);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);
	// rand_init(C, M, N);

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         m, n, k, alpha, A, k, B, n, beta, C, n);

	// double beta = 1.0;
	// cake_sgemm(A, B, C, M, N, K, p, NULL);

	cake_cntx_t* cake_cntx = cake_query_cntx();

	clock_gettime(CLOCK_REALTIME, &start);
	
	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sgemm time: %f \n", diff_t); 


	// char fname[50];
	// snprintf(fname, sizeof(fname), "results_sq");
	// FILE *fp;
	// fp = fopen(fname, "a");
	// fprintf(fp, "cake,%d,%d,%f\n",p,M,diff_t);
	// fclose(fp);

	// cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}


