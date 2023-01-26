/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.
   Copyright (C) 2014, The University of Texas at Austin
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "blis.h"

void rand_init(float* mat, int r, int c);


int main( int argc, char** argv )
{

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

//    omp_set_num_threads(p);

    // set the number of threads
	// rntm_t rntm;
	// bli_rntm_init( &rntm );
	int p = 4;
	bli_thread_set_num_threads(p);


	int Ms[] = {16,16,32,32,32,64,64,64, 64, 64, 8 ,16 ,32 ,32 ,64 ,64 ,128,128,256,256};
	int Ks[] = {27 ,144,144,288,16 ,288,576,32 , 40, 64, 27 , 8 , 16 , 32 , 32 , 64 , 64 , 128, 128, 256};
	int Ns[] = {1024,1024,256,256,256,64,64,64, 122, 125, 2304,2304,576,576,144,144, 36, 36, 9, 9};

	for(int t = 0; t < 20; t++) {

		m = Ms[t], k = Ks[t], n = Ns[t];

		rsc = 1; csc = m;
		rsa = 1; csa = m;
		rsb = 1; csb = k;

		a = (float*) malloc( m * k * sizeof( float ) );
		b = (float*) malloc( k * n * sizeof( float ) );
		c = (float*) calloc( m * n , sizeof( float ) );

		// Set the scalars to use.
		alpha = 1.0;
		beta  = 1.0;

	    srand(time(NULL));
		rand_init(a, m, k);
		rand_init(b, k, n);

	    int ntrials = atoi(argv[1]);

	    float ressss;
	    float tttmp[18];
	    int flushsz=100000;
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

			// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
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
		fprintf(fp, "blis,%d, %d,%d,%d,%d,%f\n", t+1, p,m,k,n, diff_t / ntrials);
		fclose(fp);


		// bli_sprintm( "c: after gemm", m, n, c, rsc, csc, "%4.1f", "" );

		// Free the memory obtained via malloc().
		free( a );
		free( b );
		free( c );
	}


	return 0;
}



void rand_init(float* mat, int r, int c) {
	// int MAX = 65536;
	for(int i = 0; i < r*c; i++) {
		// mat[i] = (double) i;
		// mat[i] = 1.0;
		// mat[i] =  (double) (i%MAX);
		mat[i] =  (float) rand() / RAND_MAX*2.0 - 1.0;
	}	
}