#include "cake.h"




int main( int argc, char** argv ) {
//	  run_tests();

	int M, K, N, p = 10, ntrials = 100;
	struct timespec start, end;
	double diff_t, ans;
	float *A, *B, *C;
	long seconds, nanoseconds;
    cake_cntx_t* cake_cntx = cake_query_cntx();

	char fname[50];
	snprintf(fname, sizeof(fname), "result_skew");

    FILE *fp;

	K = atoi(argv[1]);
	N = atoi(argv[2]);

	for(int i = 0; i < ntrials; i++) {
		for(M = 500; M < 20001; M += 500) {

			A = (float*) malloc(M * K * sizeof( float ));
			B = (float*) malloc(K * N * sizeof( float ));
			C = (float*) calloc(M * N , sizeof( float ));

		    srand(time(NULL));
			rand_init(A, M, K);
			rand_init(B, K, N);

		    clock_gettime(CLOCK_REALTIME, &start);

			ans = cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0,0,1,0,KMN);

		    clock_gettime(CLOCK_REALTIME, &end);
		    seconds = end.tv_sec - start.tv_sec;
		    nanoseconds = end.tv_nsec - start.tv_nsec;
		    diff_t = seconds + nanoseconds*1e-9;

		    fp = fopen(fname, "a");
		    fprintf(fp, "k-first,%d,%d,%d,%f\n",M,K,N,ans);
		    fclose(fp);
		
			free(A);
			free(B);
			free(C);





			A = (float*) malloc(M * K * sizeof( float ));
			B = (float*) malloc(K * N * sizeof( float ));
			C = (float*) calloc(M * N , sizeof( float ));

		    srand(time(NULL));
			rand_init(A, M, K);
			rand_init(B, K, N);
			
		    clock_gettime(CLOCK_REALTIME, &start);

			ans = cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0,0,1,0, MKN);

		    clock_gettime(CLOCK_REALTIME, &end);
		    seconds = end.tv_sec - start.tv_sec;
		    nanoseconds = end.tv_nsec - start.tv_nsec;
		    diff_t = seconds + nanoseconds*1e-9;

		    fp = fopen(fname, "a");
		    fprintf(fp, "m-first,%d,%d,%d,%f\n",M,K,N,ans);
		    fclose(fp);
		
			free(A);
			free(B);
			free(C);





			A = (float*) malloc(M * K * sizeof( float ));
			B = (float*) malloc(K * N * sizeof( float ));
			C = (float*) calloc(M * N , sizeof( float ));

		    srand(time(NULL));
			rand_init(A, M, K);
			rand_init(B, K, N);
			
		    clock_gettime(CLOCK_REALTIME, &start);

			ans = cake_sgemm(A, B, C, M, N, K, p, cake_cntx);

		    clock_gettime(CLOCK_REALTIME, &end);
		    seconds = end.tv_sec - start.tv_sec;
		    nanoseconds = end.tv_nsec - start.tv_nsec;
		    diff_t = seconds + nanoseconds*1e-9;

		    fp = fopen(fname, "a");
		    fprintf(fp, "opt,%d,%d,%d,%f\n",M,K,N,ans);
		    fclose(fp);
		
			free(A);
			free(B);
			free(C);
		}
	}

	free(cake_cntx);

	return 0;
}

