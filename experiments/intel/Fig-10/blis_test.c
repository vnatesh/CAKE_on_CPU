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

    int p = atoi(argv[4]);
//    omp_set_num_threads(p);

    // set the number of threads
	// rntm_t rntm;
	// bli_rntm_init( &rntm );
	bli_thread_set_num_threads(p);

	m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);

	rsc = 1; csc = m;
	rsa = 1; csa = m;
	rsb = 1; csb = k;

	a = (float*) malloc( m * k * sizeof( float ) );
	b = (float*) malloc( k * n * sizeof( float ) );
	c = (float*) calloc( m * n , sizeof( float ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize the matrix operands.
	// bli_srandm( 0, BLIS_DENSE, m, k, a, rsa, csa );

	// bli_srandm( 0, BLIS_DENSE, k, n, b, rsb, csb );
	// bli_ssetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
 //               k, n, &one, b, rsb, csb );
	// bli_ssetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
 //               m, n, &zero, c, rsc, csc );

	// bli_srandm( 0, BLIS_DENSE, m, n, c, rsc, csc );

    srand(time(NULL));
	rand_init(a, m, k);
	rand_init(b, k, n);


 //    clock_gettime(CLOCK_REALTIME, &start);

	// // c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
	// bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
	//            m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
	//                      &beta, c, rsc, csc );

 //    clock_gettime(CLOCK_REALTIME, &end);
 //    seconds = end.tv_sec - start.tv_sec;
 //    nanoseconds = end.tv_nsec - start.tv_nsec;
 //    diff_t = seconds + nanoseconds*1e-9;
 //    printf("blis sgemm 1 time: %f \n", diff_t/1); 



    clock_gettime(CLOCK_REALTIME, &start);

    for(int i = 0; i < 1; i++) {
		// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
		bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
		           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
		                     &beta, c, rsc, csc );
	}

    clock_gettime(CLOCK_REALTIME, &end);


    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
    printf("blis sgemm time: %f \n", diff_t/1); 

	// bli_sprintm( "c: after gemm", m, n, c, rsc, csc, "%4.1f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );

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