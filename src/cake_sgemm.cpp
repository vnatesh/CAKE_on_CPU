#include "cake.h"



double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


	int A_sz, B_sz, C_sz;	
	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;

	sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, sch);
	omp_set_num_threads(p);

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d\n", x->m_c, x->k_c, x->n_c);

	if(packedA) {
		A_p = A;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &A_p, 64, A_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_A(A, A_p, M, K, p, x, cake_cntx, sch);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A pack time: %f \n", diff_t ); 
	}


	if(packedB) {
		B_p = B;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

	    B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B pack time: %f \n", diff_t ); 
	}


	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
	// otherwise just allocate an empty C_p buffer
	if(beta != 0) {

		clock_gettime(CLOCK_REALTIME, &start);

	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &C_p, 64, C_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_C(C, C_p, M, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("C pack time: %f \n", diff_t ); 

	} else {
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch) / sizeof(float);
	    C_p = (float*) calloc(C_sz, sizeof(float));
	}

	clock_gettime(CLOCK_REALTIME, &start);

	schedule(A_p, B_p, C_p, M, N, K, p, cake_cntx, x, sch);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

	times = diff_t;

	clock_gettime(CLOCK_REALTIME, &start);

	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);


	if(!packedA) free(A_p);
	if(!packedB) free(B_p);
	free(C_p);

	return times;
}



double cake_sp_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


	int A_sz, B_sz, C_sz;	
	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;

	sch = KMN;
	// sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, sch);
	omp_set_num_threads(p);

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d\n", x->m_c, x->k_c, x->n_c);

	sp_pack_t* sp_pack;

	if(packedA) {
		A_p = A;
	} else {


		// print_mat(A,M,K);

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &A_p, 64, A_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}
		// A_sz = cake_sgemm_packed_A_size(M, K, p, cake_cntx) / sizeof(float);
	 //    A_p = (float*) calloc(A_sz, sizeof(float));

		sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));

		pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A sparse pack time: %f \n", diff_t ); 

		// free(A_p);
		// A_p = sp_pack->A_sp_p;

		// for(int i = 0; i < x->M_padded*K; i++) {
		// 	printf("%f ", A_p[i]);
		// }

		// clock_gettime(CLOCK_REALTIME, &start);

		// pack_A(A, A_p, M, K, p, x, cake_cntx, sch);

		// clock_gettime(CLOCK_REALTIME, &end);
		// seconds = end.tv_sec - start.tv_sec;
		// nanoseconds = end.tv_nsec - start.tv_nsec;
		// diff_t = seconds + nanoseconds*1e-9;
		// if(DEBUG) printf("A dense pack time: %f \n", diff_t ); 

		// int num_obs = x->m_pad ? (x->Mb-1)*p*x->Kb + x->p_l*x->Kb : x->Mb*p*x->Kb;

		// printf("nnz_outer_blk\n");
		// for(int i = 0; i < ((x->M_padded*x->Kb) / cake_cntx->mr); i++) {
		// 	printf("%d ", sp_pack->nnz_outer_blk[i]);
		// }
		// printf("\n\n");


		// printf("nnz_outer\n");
		// for(int i = 0; i < x->M_padded*K/cake_cntx->mr; i++) {
		// 	printf("%d ", sp_pack->nnz_outer[i]);
		// }
		// printf("\n\n");


		// printf("loc_m\n");
		// for(int i = 0; i < x->M_padded*K; i++) {
		// 	printf("%d ", sp_pack->loc_m[i]);
		// }
		// printf("\n\n");

	}

		// exit(1);


	if(packedB) {
		B_p = B;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

	    B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
		if(posix_memalign((void**) &B_p, 64, B_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B pack time: %f \n", diff_t ); 
	}


	// C = alpha*A*B + beta*C. If beta is !=0, we must explicitly pack C, 
	// otherwise just allocate an empty C_p buffer
	if(beta != 0) {
		clock_gettime(CLOCK_REALTIME, &start);
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &C_p, 64, C_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_C(C, C_p, M, N, p, x, cake_cntx, sch);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("C pack time: %f \n", diff_t ); 

	} else {
	    C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch) / sizeof(float);
	    C_p = (float*) calloc(C_sz, sizeof(float));
	}

	clock_gettime(CLOCK_REALTIME, &start);

	schedule_sp(sp_pack, B_p, C_p, M, N, K, p, cake_cntx, x, sch);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);
	// print_packed_C(C_p, M, N, m_c, n_c);
	// unpack_C(C, C_p, M, N, m_c, n_c, n_r, m_r, p);
	times = diff_t;

	clock_gettime(CLOCK_REALTIME, &start);

	// unpack_C_rsc(C, C_p, M, N, m_c, n_c, n_r, m_r, p, alpha_n); 
	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);


	if(!packedA) free(A_p);
	if(!packedB) free(B_p);
	free(C_p);

	return times;
}


void schedule_sp(sp_pack_t* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch) {

	switch(sch) {
		case KMN: {
			schedule_KMN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case MKN: {
			// schedule_MKN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case NKM: {
			// schedule_NKM_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}

void schedule(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch) {

	switch(sch) {
		case KMN: {
			schedule_KMN(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case MKN: {
			schedule_MKN(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case NKM: {
			schedule_NKM(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}


enum sched set_schedule(enum sched sch, int M, int N, int K) {

	if(sch == NA) {
		if(M >= 2*K && M >= N)  {
		    if(DEBUG) printf("MKN Cake Schedule\n");
			sch = MKN;
		} else if(K >= M && K >= N) {
		    if(DEBUG) printf("KMN Cake Schedule\n");
			sch = KMN;
		} else if(N >= 2*K && N >= M) {
			if(DEBUG) printf("NKM Cake Schedule\n");
			sch = NKM;
		} else {
			if(DEBUG) printf("KMN Cake Schedule\n");
			sch = KMN;
		}
	}

	return sch;
}



