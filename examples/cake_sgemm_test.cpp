

#include "cake.h"



int main( int argc, char** argv ) {
	 // run_tests();

    if(argc < 3) {
        printf("Enter M, K, and N\n");
        exit(1);
    }

	int M, K, N, p;
	struct timespec start, end;
    long seconds, nanoseconds;
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
		update_mr_nr(cake_cntx, 20, 96);

//	cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
	
	clock_gettime(CLOCK_REALTIME, &start);

	// cake_sp_sgemm(A, B, C, M, N, K, p, cake_cntx, 1-sparsity);

    clock_gettime(CLOCK_REALTIME, &end);
     seconds = end.tv_sec - start.tv_sec;
     nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	printf("sp_sgemm time: %f \n", diff_t); 




	update_mr_nr(cake_cntx, 6, 16);

    int ntrials = atoi(argv[5]);
    float ressss;
    float tttmp[18];
    int flushsz=100000000;
    diff_t = 0.0;


    for(int i = 0; i < ntrials; i++) {


        float *dirty = (float *)malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){
            dirty[dirt] += dirt%100;
            tttmp[dirt%18] += dirty[dirt];
        }

        for(int ii =0; ii<18;ii++){
            ressss+= tttmp[ii];
        }


		clock_gettime(CLOCK_REALTIME, &start);

		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
		// cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
		// cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		cake_sgemm_online_test(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// cake_sgemm_online_test(A, B, C, M, N, K, p, cake_cntx);

		// cake_sgemm_small_test(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);

	    clock_gettime(CLOCK_REALTIME, &end);
         seconds = end.tv_sec - start.tv_sec;
         nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
	    printf("cake_sgemm_online_test time: %f \n", diff_t); 


		clock_gettime(CLOCK_REALTIME, &start);

		cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

		
	    clock_gettime(CLOCK_REALTIME, &end);
         seconds = end.tv_sec - start.tv_sec;
         nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
	    printf("cake_sgemm time: %f \n", diff_t); 


        free(dirty);
    }


    printf("cake_sgemm time: %f \n", diff_t / ntrials); 

	cake_sgemm_checker(A, B, C, N, M, K);
	
	free(A);
	free(B);
	free(C);
	free(cake_cntx);
	
	return 0;
}




