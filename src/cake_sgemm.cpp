#include "cake.h"




double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);

	// if(cake_gemm_small(A, B, C, M, N, K, p, x, cake_cntx, sch)) {
	// 	return 1;
	// }

	size_t A_sz, B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t;
	float *A_p, *B_p, *C_p;


	clock_gettime(CLOCK_REALTIME, &start1);

	init_block_dims(M, N, K, p, x, cake_cntx, sch, argv, 0);
	sch = x->sch;

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);
    if(DEBUG) print_schedule(sch);

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

	schedule(A_p, B_p, C_p, M, N, K, p, cake_cntx, x, sch, 0, 0);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

	clock_gettime(CLOCK_REALTIME, &start);

	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);

    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);

	if(!packedA) free(A_p);
	if(!packedB) free(B_p);
	free(C_p);
	free(x);

	return diff_t;
}



double cake_sp_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[], 
	bool packedA, sp_pack_t* sp_pack, bool packedB, 
	float alpha, float beta, enum sched sch) {


	size_t A_sz, B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;

	sch = KMN;
	// sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));


	clock_gettime(CLOCK_REALTIME, &start1);

	init_block_dims(M, N, K, p, x, cake_cntx, sch, argv, density);
	omp_set_num_threads(p);

	if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
	if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);


	if(sp_pack == NULL) {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch) / sizeof(float);
	    A_p = (float*) calloc(A_sz, sizeof(float));

		sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));

		pack_A_sp(A, A_p, M, K, p, sp_pack, x, cake_cntx, sch);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A sparse pack time: %f \n", diff_t ); 
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



    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);


	if(!packedA) {
		free(sp_pack->loc_m); 
		free(sp_pack->nnz_outer); 
		free(sp_pack->k_inds); 
		free(sp_pack->A_sp_p);
		free(sp_pack);
	}

	if(!packedB) free(B_p);
	free(C_p);
	free(x);

	return times;
}







double cake_sp_sgemm_compressed(char* fname, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, float density, char* argv[], sp_pack_t* sp_pack,
	bool packedA, bool packedB, float alpha, float beta, enum sched sch, int alg) {


	size_t B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *B_p, *C_p;
	csr_t* csr;

	sch = KMN;
	// sch = set_schedule(sch, M, N, K);

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, sch, argv, density, 4, alg);
	omp_set_num_threads(p);

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);

	if(sp_pack == NULL) {

		csr = file_to_csr(fname);
		sp_pack_t* sp_pack = malloc_sp_pack(M, K, csr->rowptr[M], x, cake_cntx);
		pack_A_csr_to_sp_k_first(csr, M, K, csr->rowptr[M], p, sp_pack, x, cake_cntx);
		free_csr(csr);
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

	schedule_KMN_sp_test(sp_pack, B_p, C_p, M, N, K, p, cake_cntx, x);

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



    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);


	if(!packedA) {
		free_sp_pack(sp_pack);
	}

	if(!packedB) free(B_p);

	free(C_p);
	free(x);

	return times;
}




double cake_sgemm_2d(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);

	// if(cake_gemm_small(A, B, C, M, N, K, p, x, cake_cntx, sch)) {
	// 	return 1;
	// }

	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p[p];

	sch = KMN;

	init_block_dims_2d(M, N, K, p, x, cake_cntx, sch, argv, 0);
	sch = x->sch;

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);
    if(DEBUG) print_schedule(sch);

	if(posix_memalign((void**) &A_p, 64, x->pm * x->m_c * x->k_c * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}
	

	if(posix_memalign((void**) &B_p, 64, x->pn * x->n_c * x->k_c * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}


	for(int i = 0; i < p; i++) {
		C_p[i] = (float*) calloc(x->m_c * x->n_c, sizeof(float));
	}
	
	clock_gettime(CLOCK_REALTIME, &start);

	schedule_KMN_2d(A, B, C, A_p, B_p, C_p, M, N, K, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

	times = diff_t;

	free(A_p);
	for(int i = 0; i < p; i++) {
		free(C_p[i]);
	}
	free(B_p);
	free(x);

	return times;
}





double cake_sgemm_2d_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);

	// if(cake_gemm_small(A, B, C, M, N, K, p, x, cake_cntx, sch)) {
	// 	return 1;
	// }

	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p[p];

	sch = KMN;

	init_block_dims_2d_small(M, N, K, p, x, cake_cntx, sch, argv, 0);
	sch = x->sch;

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);
    if(DEBUG) print_schedule(sch);

	if(posix_memalign((void**) &A_p, 64, x->M_padded * x->k_c * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}
	

	if(posix_memalign((void**) &B_p, 64, x->k_c * x->N_padded * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}


	for(int i = 0; i < p; i++) {
		C_p[i] = (float*) calloc(x->m_c * x->n_c, sizeof(float));
	}
	
	clock_gettime(CLOCK_REALTIME, &start);

	schedule_KMN_2d_small(A, B, C, A_p, B_p, C_p, M, N, K, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

	times = diff_t;

	free(A_p);
	for(int i = 0; i < p; i++) {
		free(C_p[i]);
	}
	free(B_p);
	free(x);

	return times;
}



// double cake_sgemm_2d(float* A, float* B, float* C, int M, int N, int K, int p, 
// 	cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


// 	if(cake_cntx == NULL) {
// 		cake_cntx = cake_query_cntx();
// 	}

// 	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
// 	omp_set_num_threads(p);

// 	// if(cake_gemm_small(A, B, C, M, N, K, p, x, cake_cntx, sch)) {
// 	// 	return 1;
// 	// }

// 	size_t A_sz, B_sz, C_sz;	
// 	struct timespec start, end, start1, end1;
// 	long seconds, nanoseconds;
// 	double diff_t, times;
// 	float *A_p[p], *B_p[p], *C_p[p];

// 	sch = KMN;

// 	clock_gettime(CLOCK_REALTIME, &start1);

// 	init_block_dims_2d(M, N, K, p, x, cake_cntx, sch, argv, 0);
// 	sch = x->sch;

//     if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
//     if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);
//     if(DEBUG) print_schedule(sch);

// 	for(int i = 0; i < p; i++) {
// 		if(posix_memalign((void**) &A_p[i], 64, x->m_c * x->k_c * sizeof(float))) {
// 			printf("posix memalign error\n");
// 			exit(1);
// 		}
// 		// A_p[i] = (float*) malloc(m_c * k_c * sizeof(float));
// 	}

// 	for(int i = 0; i < p; i++) {
// 		if(posix_memalign((void**) &B_p[i], 64, x->k_c * x->n_c * sizeof(float))) {
// 			printf("posix memalign error\n");
// 			exit(1);
// 		}
// 	}

// 	for(int i = 0; i < p; i++) {
// 		C_p[i] = (float*) calloc(x->m_c * x->n_c, sizeof(float));
// 	}
	
// 	clock_gettime(CLOCK_REALTIME, &start);

// 	schedule_KMN_2d(A, B, C, A_p, B_p, C_p, M, N, K, p, cake_cntx, x);

//     clock_gettime(CLOCK_REALTIME, &end);
//     seconds = end.tv_sec - start.tv_sec;
//     nanoseconds = end.tv_nsec - start.tv_nsec;
//     diff_t = seconds + nanoseconds*1e-9;
// 	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

// 	times = diff_t;


// 	for(int i = 0; i < p; i++) {
// 		free(A_p[i]);
// 		free(C_p[i]);
// 		free(B_p[i]);
// 	}
// 	free(x);

// 	return times;
// }


double cake_sgemm_online(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB, float alpha, float beta, enum sched sch) {


	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);

	// if(cake_gemm_small(A, B, C, M, N, K, p, x, cake_cntx, sch)) {
	// 	return 1;
	// }

	struct timespec start, end;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p[p], *B_p, *C_p[p];

	sch = KMN;

	init_block_dims(M, N, K, p, x, cake_cntx, sch, argv, 0);
	sch = x->sch;

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);
    if(DEBUG) print_schedule(sch);


	for(int i = 0; i < p; i++) {
		if(posix_memalign((void**) &A_p[i], 64, x->m_c * x->k_c * sizeof(float))) {
			printf("posix memalign error\n");
			exit(1);
		}
		// A_p[i] = (float*) malloc(m_c * k_c * sizeof(float));
	}

	if(posix_memalign((void**) &B_p, 64, x->k_c * x->n_c * sizeof(float))) {
		printf("posix memalign error\n");
		exit(1);
	}
	// float* B_p = (float*) malloc(k_c * n_c * sizeof(float));

	for(int i = 0; i < p; i++) {
		C_p[i] = (float*) calloc(x->m_c * x->n_c, sizeof(float));
	}
	
	clock_gettime(CLOCK_REALTIME, &start);

	schedule_KMN_online(A, B, C, A_p, B_p, C_p, M, N, K, p, cake_cntx, x);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

	times = diff_t;


	for(int i = 0; i < p; i++) {
		free(A_p[i]);
		free(C_p[i]);
	}
	free(B_p);
	free(x);

	return times;
}



double cake_sgemm_test(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[], bool packedA, bool packedB, float alpha, float beta, enum sched sch) {

	sch = KMN;

	if(cake_cntx == NULL) {
		cake_cntx = cake_query_cntx();
	}

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	omp_set_num_threads(p);

	// if(cake_gemm_small(A, B, C, M, N, K, p, x, cake_cntx, sch)) {
	// 	return 1;
	// }

	size_t A_sz, B_sz, C_sz;	
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t, times;
	float *A_p, *B_p, *C_p;


	clock_gettime(CLOCK_REALTIME, &start1);

	init_block_dims(M, N, K, p, x, cake_cntx, sch, argv, 0);
	sch = x->sch;

    if(DEBUG) printf("m_r = %d, n_r = %d\n\n", cake_cntx->mr, cake_cntx->nr);
    if(DEBUG) printf("mc = %d, kc = %d, nc = %d, alpha_n = %f\n", x->m_c, x->k_c, x->n_c, cake_cntx->alpha_n);
    if(DEBUG) print_schedule(sch);

	if(packedA) {
		A_p = A;
	} else {

		clock_gettime(CLOCK_REALTIME, &start);

		A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);
		if(posix_memalign((void**) &A_p, 64, A_sz)) {
			printf("posix memalign error\n");
			exit(1);
		}

		pack_A_single_buf_k_first_blis(A, A_p, M, K, p, x, cake_cntx);

		clock_gettime(CLOCK_REALTIME, &end);
		seconds = end.tv_sec - start.tv_sec;
		nanoseconds = end.tv_nsec - start.tv_nsec;
		diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("A pack time blis: %f \n", diff_t ); 
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

		pack_B_k_first_blis(B, B_p, K, N, x, cake_cntx);

	    clock_gettime(CLOCK_REALTIME, &end);
	    seconds = end.tv_sec - start.tv_sec;
	    nanoseconds = end.tv_nsec - start.tv_nsec;
	    diff_t = seconds + nanoseconds*1e-9;
		if(DEBUG) printf("B pack time blis: %f \n", diff_t ); 
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

	// schedule_KMN_C_unpacked(A_p, B_p, C_p, M, N, K, p, cake_cntx, x);
	schedule(A_p, B_p, C_p, M, N, K, p, cake_cntx, x, sch, 0, 0);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("GEMM time: %f \n", diff_t); 	// exit(1);

	times = diff_t;

	clock_gettime(CLOCK_REALTIME, &start);

	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 
    // cake_sgemm_checker(A, B, C_p, N, M, K);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);

    clock_gettime(CLOCK_REALTIME, &end1);
    seconds = end1.tv_sec - start1.tv_sec;
    nanoseconds = end1.tv_nsec - start1.tv_nsec;
    diff_t = seconds + nanoseconds*1e-9;
	if(DEBUG) printf("full gemm time: %f \n", diff_t); 	// exit(1);

	if(!packedA) free(A_p);
	if(!packedB) free(B_p);
	free(C_p);
	free(x);


	return times;
}




bool cake_gemm_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, char* argv[]) {

	// size_t A_sz, B_sz, C_sz;	
	// struct timespec start, end;
	// long seconds, nanoseconds;
	// double diff_t;
	// float *A_p, *B_p, *C_p;
	// // use packing-free kernels if packing time is more than 
	// // x% (10% here) of the projected total runtime
	// double t_pack = ((double) (4*2*(M*K + K*N + M*N))) / cake_cntx->peak_dram_bw; // read and write, 4 byte elements
	// double t_comp = ((double) M*N*K) / (0.8*cake_cntx->peak_flops); // assume we can only reach 80% of peak
	// double pack_thresh = t_pack / (t_pack + t_comp);

	// if(pack_thresh < 0.10) {
	// 	return 0;
	// }

	// double t_pack_a = ((double) 4.0*2*M*K) / cake_cntx->peak_dram_bw;
	// double t_pack_b = ((double) 4.0*2*N*K) / cake_cntx->peak_dram_bw;
	// double t_pack_c = ((double) 4.0*2*M*N) / cake_cntx->peak_dram_bw;

	// if(t_pack_a / (t_pack + t_comp) < 0.05 ) {

	// 	sch = KMN;
	// 	init_block_dims(M, N, K, p, x, cake_cntx, sch,  argv, 0);
	// 	schedule(A, B, C, M, N, K, p, cake_cntx, x, sch, 0, 1);

	// 	// sch = NKM;
	// 	// init_block_dims(M, N, K, p, x, cake_cntx, sch,  argv, 0);

	// 	// clock_gettime(CLOCK_REALTIME, &start);

	// 	// A_sz = cake_sgemm_packed_A_size(M, K, p, x, cake_cntx, sch);
	// 	// if(posix_memalign((void**) &A_p, 64, A_sz)) {
	// 	// 	printf("posix memalign error\n");
	// 	// 	exit(1);
	// 	// }
	// 	// pack_A(A, A_p, M, K, p, x, cake_cntx, sch);

	// 	// clock_gettime(CLOCK_REALTIME, &end);
	// 	// seconds = end.tv_sec - start.tv_sec;
	// 	// nanoseconds = end.tv_nsec - start.tv_nsec;
	// 	// diff_t = seconds + nanoseconds*1e-9;
	// 	// if(DEBUG) printf("A pack time: %f \n", diff_t ); 

	// 	// schedule_NKM_small_A_packed(A_p, B, C, M, N, K, p, cake_cntx, x);
	// 	// free(A_p);

	// } else 
	// if(t_pack_b / (t_pack + t_comp) < 0.05) {

	// 	sch = MKN;
	// 	init_block_dims(M, N, K, p, x, cake_cntx, sch,  argv, 0);

	// 	clock_gettime(CLOCK_REALTIME, &start);

	//     B_sz = cake_sgemm_packed_B_size(K, N, p, x, cake_cntx);
	// 	if(posix_memalign((void**) &B_p, 64, B_sz)) {
	// 		printf("posix memalign error\n");
	// 		exit(1);
	// 	}

	// 	pack_B(B, B_p, K, N, p, x, cake_cntx, sch);

	//     clock_gettime(CLOCK_REALTIME, &end);
	//     seconds = end.tv_sec - start.tv_sec;
	//     nanoseconds = end.tv_nsec - start.tv_nsec;
	//     diff_t = seconds + nanoseconds*1e-9;
	// 	if(DEBUG) printf("B pack time: %f \n", diff_t ); 

	// 	schedule_MKN_small_B_packed(A, B_p, C, M, N, K, p, cake_cntx, x);
	// 	free(B_p);

	// } else if(t_pack_c / (t_pack + t_comp) < 0.05) {

	// 	sch = KMN;
	// 	init_block_dims(M, N, K, p, x, cake_cntx, sch,  argv, 0);

	//     C_sz = cake_sgemm_packed_C_size(M, N, p, x, cake_cntx, sch) / sizeof(float);
	//     C_p = (float*) calloc(C_sz, sizeof(float));

	// 	schedule_KMN_small(A, B, C_p, M, N, K, p, cake_cntx, x);

	// 	clock_gettime(CLOCK_REALTIME, &start);

	// 	unpack_C(C, C_p, M, N, p, x, cake_cntx, sch); 

	//     clock_gettime(CLOCK_REALTIME, &end);
	//     seconds = end.tv_sec - start.tv_sec;
	//     nanoseconds = end.tv_nsec - start.tv_nsec;
	//     diff_t = seconds + nanoseconds*1e-9;
	// 	if(DEBUG) printf("unpacking time: %f \n", diff_t); 	// exit(1);

	// 	free(C_p);

	// } else 
	// if(sch == NKM) {
	// 	sch = KMN;
	// 	init_block_dims(M, N, K, p, x, cake_cntx, sch,  argv, 0);
	// 	schedule(A, B, C, M, N, K, p, cake_cntx, x, sch, 0, 1);
	// } else {
	// 	init_block_dims(M, N, K, p, x, cake_cntx, sch,  argv, 0);
	// 	schedule(A, B, C, M, N, K, p, cake_cntx, x, sch, 0, 1);		
	// }

	return 1;
}

// mv results shmoo; scp shmoo vikas@10.0.0.185:/Users/vikas/Documents/test

void schedule_sp(sp_pack_t* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch) {

	switch(sch) {
		case KMN: {
			schedule_KMN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case MKN: {
			schedule_MKN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		case NKM: {
			schedule_NKM_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}


void schedule(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch, bool sparse, bool small) {

	switch(sch) {
		case KMN: {
			if(sparse) {
				// schedule_KMN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x, 1, 0); 
			} else if(small) {
				// schedule_KMN_small(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			} else {
				schedule_KMN(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			}
			break;
		}
		case MKN: {
			if(sparse) {
				// schedule_KMN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x, 1, 0); 
			} else if(small) {
				// schedule_MKN_small(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			} else {
				schedule_MKN(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			}
			break;
		}
		case NKM: {
			if(sparse) {
				// schedule_KMN_sp(A_p, B_p, C_p, M, N, K, p, cake_cntx, x, 1, 0); 
			} else if(small) {
				// schedule_NKM_small(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			} else {
				schedule_NKM(A_p, B_p, C_p, M, N, K, p, cake_cntx, x); 
			}
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

