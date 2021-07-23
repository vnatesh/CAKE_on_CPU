#include "cake.h"

 
int main( int argc, char** argv ) {

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	p = 10;

	printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

	float* A = (float*) malloc(M * K * sizeof( float ));
	float* B = (float*) malloc(K * N * sizeof( float ));
	float* C = (float*) calloc(M * N , sizeof( float ));

	// initialize A and B
    srand(time(NULL));
	rand_init(A, M, K);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
	cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);

	return 0;
}


