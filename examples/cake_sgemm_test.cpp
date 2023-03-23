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
	rand_init(A, M, K);
	rand_init(B, K, N);

	cake_cntx_t* cake_cntx = cake_query_cntx();
	int ms[] = {8,16,32};
	int ns[] = {12,24,48,72,96};

    cake_cntx->mr = ms[0];
    cake_cntx->nr = ns[0];
    cake_cntx->m_map = 0;
    cake_cntx->n_map = 0;

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


		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
		// cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
		// cake_sgemm(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// cake_sgemm_online_test(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// cake_sgemm_online_test(A, B, C, M, N, K, p, cake_cntx);
		// diff_t += cake_sgemm_blis(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		// cake_sgemm_small_test(A, B, C, M, N, K, p, cake_cntx, argv, 0, 0, 1, 0, KMN);
		diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN);
		// diff_t += cake_sgemm_2d(A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN);


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





















// #include "cake.h"



// typedef double cake_sgemm_tester(float* A, float* B, float* C, int M, int N, int K, int p, 
//     cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB , 
//     float alpha, float beta, enum sched sch, int mcu, int kcu, int ncu);


// static cake_sgemm_tester* test_funcs[2] = 
// {
//     cake_sgemm,
//     cake_sgemm_online
// };



// void square_test(int func, int ntrials, int p, int s, int e, int step) {

//     int M, K, N;
//     struct timespec start, end;
//     long seconds, nanoseconds;
//     double diff_t;

//     for(int t = s; t < (e + 1); t += step) {

//         M = t;
//         K = t;
//         N = t;

//         printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

//         float* A = (float*) malloc(M * K * sizeof( float ));
//         float* B = (float*) malloc(K * N * sizeof( float ));
//         float* C = (float*) calloc(M * N , sizeof( float ));

//         // initialize A and B
//         srand(time(NULL));
//         rand_init(A, M, K);
//         rand_init(B, K, N);

//         cake_cntx_t* cake_cntx = cake_query_cntx();


//         float ressss;
//         float tttmp[18];
//         int flushsz=2*cake_cntx->L3 / sizeof(float);
//         diff_t = 0.0;


//         for(int i = 0; i < ntrials; i++) {


//             float *dirty = (float *)malloc(flushsz * sizeof(float));
//             #pragma omp parallel for
//             for (int dirt = 0; dirt < flushsz; dirt++){
//                 dirty[dirt] += dirt%100;
//                 tttmp[dirt%18] += dirty[dirt];
//             }

//             for(int ii =0; ii<18;ii++){
//                 ressss+= tttmp[ii];
//             }

//             diff_t += test_funcs[func](A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN, 0, 0, 0);

//             free(dirty);
//         }


//         // printf("cake_sgemm time: %f \n", diff_t / ntrials); 
//         // int write_result = atoi(argv[13]);
//         // if(write_result) {
//         char fname[50];
//         snprintf(fname, sizeof(fname), "results");
//         FILE *fp;
//         fp = fopen(fname, "a");
//         fprintf(fp, "%d,%d,%d,%d,%d,%f\n", func, p, M, K, N, diff_t / ntrials);
//         fclose(fp);
//         // }


        
//         free(A);
//         free(B);
//         free(C);
//         free(cake_cntx);
    
//     }
// }



// int main( int argc, char** argv ) {

//     // run_tests();

//     int ntrials = atoi(argv[1]);
//     int p = atoi(argv[2]);
//     int start = atoi(argv[3]);
//     int end = atoi(argv[4]);
//     int step = atoi(argv[5]);

//     for(int i = 0; i < 2; i++) {
//         square_test(i, ntrials, p, start, end, step);
//     }

//     return 0;
// }


