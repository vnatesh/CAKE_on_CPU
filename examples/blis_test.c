#include "blis.h"

void rand_init(float* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
	}	
}

void square_test(int func, int ntrials, int p, int s, int e, int step) {

    struct timespec start, end;
    long seconds, nanoseconds;
    double diff_t;

	dim_t m, n, k;
	inc_t rsa, csa;
	inc_t rsb, csb;
	inc_t rsc, csc;

	float* a;
	float* b;
	float* c;
	float  alpha, beta;

	bli_thread_set_num_threads(p);

	for(int t = s; t < (e + 1); t += step) {

		m = t, k = t, n = t;

		printf("M = %d, K = %d, N = %d, cores = %d\n", m,k,n,p);

		// rsc = 1; csc = m;
		// rsa = 1; csa = m;
		// rsb = 1; csb = k;
		rsc = n; csc = 1;
		rsa = k; csa = 1;
		rsb = n; csb = 1;

		a = (float*) malloc( m * k * sizeof( float ) );
		b = (float*) malloc( k * n * sizeof( float ) );
		c = (float*) calloc( m * n , sizeof( float ) );

		// Set the scalars to use.
		alpha = 1.0;
		beta  = 1.0;

	    srand(time(NULL));
		rand_init(a, m, k);
		rand_init(b, k, n);

	    float ressss;
	    float tttmp[18];
	    int flushsz=100000000; // 400 MB of flaots
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

			bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
			           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
			                     &beta, c, rsc, csc );

			clock_gettime(CLOCK_REALTIME, &end);
			seconds = end.tv_sec - start.tv_sec;
			nanoseconds = end.tv_nsec - start.tv_nsec;
			diff_t += seconds + nanoseconds*1e-9;

	        free(dirty);
		}


		char fname[50];
		snprintf(fname, sizeof(fname), "results");
		FILE *fp;
		fp = fopen(fname, "a");
		fprintf(fp, "%d,%d,%d,%d,%d,%f\n", func, p, m, k, n, diff_t / ntrials);
		fclose(fp);

		// bli_sprintm( "c: after gemm", m, n, c, rsc, csc, "%4.1f", "" );

		// Free the memory obtained via malloc().
		free( a );
		free( b );
		free( c );
	}
}


int main( int argc, char** argv ) {

	int ntrials = atoi(argv[1]);
	int p = atoi(argv[2]);
	int start = atoi(argv[3]);
	int end = atoi(argv[4]);
	int step = atoi(argv[5]);

	square_test(4, ntrials, p, start, end, step);
	

	return 0;
}

