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
    FILE *fp;

	dim_t m, n, k;
	inc_t rsa, csa;
	inc_t rsb, csb;
	inc_t rsc, csc;

	float* a;
	float* b;
	float* c;
	float  alpha, beta;

	char fname[50];
	snprintf(fname, sizeof(fname), "result_skew");

    int p = atoi(argv[3]), ntrials = atoi(argv[4]);
	bli_thread_set_num_threads(p);

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;


	k = atoi(argv[1]);;
	n = atoi(argv[2]);;

    srand(time(NULL));

	for(m = 500; m < 15001; m += 500) {
		for(int i = 0; i < ntrials; i++) {

			rsc = 1; csc = m;
			rsa = 1; csa = m;
			rsb = 1; csb = k;

			a = (float*) malloc( m * k * sizeof( float ) );
			b = (float*) malloc( k * n * sizeof( float ) );
			c = (float*) calloc( m * n , sizeof( float ) );

			rand_init(a, m, k);
			rand_init(b, k, n);

		    clock_gettime(CLOCK_REALTIME, &start);

			bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
			           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
			                     &beta, c, rsc, csc );

		    clock_gettime(CLOCK_REALTIME, &end);

		    seconds = end.tv_sec - start.tv_sec;
		    nanoseconds = end.tv_nsec - start.tv_nsec;
		    diff_t = seconds + nanoseconds*1e-9;
		    fp = fopen(fname, "a");
		    fprintf(fp, "blis,%d,%d,%d,%f\n",m,k,n,diff_t);
		    fclose(fp);

			free( a );
			free( b );
			free( c );
		}
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