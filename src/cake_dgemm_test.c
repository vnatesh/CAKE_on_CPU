#include "cake.h"
 
int main( int argc, char** argv ) {

	struct timeval start, end;
	double diff_t;

    if(argc < 2) {
        printf("Enter number of threads\n");
        exit(1);
    }

	int M, K, N, p;

	// M = 1111;
	// K = 1111;
	// N = 2880;
	M = 111;
	K = 111;
	N = 111;

	// M = 96;
	// K = 14583;
	// N = 96;
	// m_c = 96;
	// k_c = 96;

	// M = 23040;
	// K = 23040;
	// N = 23040;

	// M = 960;
	// K = 960;
	// N = 960;

    p = atoi(argv[1]);
	printf("M = %d, K = %d, N = %d\n", M,K,N);

	double* A = (double*) malloc(M * K * sizeof( double ));
	double* B = (double*) malloc(K * N * sizeof( double ));
	double* C = (double*) calloc(M * N , sizeof( double ));

	// initialize A and B
	gettimeofday (&start, NULL);
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);
	// rand_init(C, M, N);
	gettimeofday (&end, NULL);
	diff_t = (((end.tv_sec - start.tv_sec)*1000000L
	+end.tv_usec) - start.tv_usec) / (1000000.0);
	if(DEBUG) printf("init time: %f \n", diff_t); 

    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         m, n, k, alpha, A, k, B, n, beta, C, n);

	// double beta = 1.0;
	cake_dgemm(A, B, C, M, N, K, p);
	// bli_dprintm( "C: ", M, N, C, N, 1, "%4.4f", "" );
	cake_dgemm_checker(A, B, C, N, M, K);

	free(A);
	free(B);
	free(C);

	return 0;
}


