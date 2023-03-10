#include "cake.h"


#define kc_step 50
#define mc_step 6
#define nc_step 16
 


// intial cache dims with cake tiling params and K-first scheduling
cache_dims_t* cake_init_sa(int M, int K, int N, int p, cake_cntx_t* cake_cntx) {

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims(M, N, K, p, x, cake_cntx, KMN, NULL);
	cache_dims_t* best = (cache_dims_t*) malloc(sizeof(cache_dims_t));
	best->m_c = x->m_c;
	best->k_c = x->k_c;
	best->n_c = x->n_c;
	best->sch = KMN;
	free(x);
	return best;
}


cache_dims_t* cake_init_sa_2d(int M, int K, int N, int p, cake_cntx_t* cake_cntx) {

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	init_block_dims_2d(M, N, K, p, x, cake_cntx, KMN, NULL);
	cache_dims_t* best = (cache_dims_t*) malloc(sizeof(cache_dims_t));
	best->m_c = x->m_c;
	best->k_c = x->k_c;
	best->n_c = x->n_c;
	best->sch = KMN;
	free(x);
	return best;
}


// https://github.com/Harvard-CS249R-Fall2020/assignments/tree/master/person_detection
// https://github.com/srivatsankrishnan/cs249-assignment2/blob/master/hw_deployment/person_detection_arducam_5mp_plus/arduino_image_provider.cpp

double cake_reg_test(float* A, float* B, float* C, 
	int M, int K, int N, int mc, int kc, int nc, int p, int ntrials) {
	 // run_tests();

	struct timespec start, end;
	double diff_t;
	printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);
	cake_cntx_t* cake_cntx = cake_query_cntx();

    float ressss;
    float tttmp[18];
    int flushsz=cake_cntx->L3 / sizeof(float);
    diff_t = 0.0;
    
  //   for(int i = 0; i < ntrials; i++) {
		// // diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx);
		// diff_t += cake_sgemm_online(A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN, 54, 512, 528);
  //   }

	cache_dims_t* ss = cake_init_sa(M, K, N, p, cake_cntx);
	// cache_dims_t* ss = cake_init_sa_2d(M, K, N, p, cake_cntx);
	printf("mc = %d, kc = %d, nc = %d\n", ss->m_c, ss->k_c, ss->n_c);

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

		// diff_t += cake_sgemm_online(A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN);
		diff_t += cake_sgemm_online(A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN, mc, kc, nc);
		// diff_t += cake_sgemm_2d(A, B, C, M, N, K, p, cake_cntx);
		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx, NULL, 0, 0, 1, 0, KMN, mc, kc, nc);

        free(dirty);
    }

    printf("cake_sgemm time: %f \n", diff_t / ntrials); 
	// cake_sgemm_checker(A, B, C, N, M, K);
	free(cake_cntx);
	return diff_t / ntrials;
}



double cake_run(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, int ntrials, int m_c, int k_c, int n_c, enum sched sch) {

    float ressss;
    float tttmp[18];
    int flushsz=cake_cntx->L3 / sizeof(float);
	double diff_t = 0.0;

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
		// diff_t += cake_sgemm(A, B, C, M, N, K, p, cake_cntx, 
		// 	NULL, 0, 0, 1, 0, KMN, m_c, k_c, n_c);
		diff_t += cake_sgemm_online(A, B, C, M, N, K, p, cake_cntx, 
			NULL, 0, 0, 1, 0, KMN, m_c, k_c, n_c);

        free(dirty);
    }

    return diff_t / ntrials;
}




void assign_cache_dims(cache_dims_t* a, cache_dims_t* b) {
	a->m_c = b->m_c;
	a->k_c = b->k_c;
	a->n_c = b->n_c;
	a->sch = b->sch;
}


int clamp_val_int(int d, int min, int max) {
  const int t = d < min ? min : d;
  return t > max ? max : t;
}



int get_random_step(int step) {
	float a = normalRandom();
	int ret = (a < 0) ? ((int) floor((double) a)) : ((int) ceil((double) a));
	return ret * step;
}


// candidate = curr + randn(len(bounds)) * step_size
void get_candidate(int M, int K, int N, int p, cake_cntx_t* cake_cntx,
	cache_dims_t* cand, cache_dims_t* curr) {

	int mr = cake_cntx->mr, nr = cake_cntx->nr;

	int mc_max = (M/p) + (mr - ((M/p) % mr));
	// int mc_max = M + (mr - (M % mr));
	int kc_max = K;
	int nc_max = N + (nr - (N % nr));

	// printf("bef1 %d %d %d\n", curr->m_c, curr->k_c, curr->n_c);

	int mcu = curr->m_c + get_random_step(mc_step);
	int kcu = curr->k_c + get_random_step(kc_step);
	int ncu = curr->n_c + get_random_step(nc_step);

	// printf("%d\n", curr->m_c +   normalRandom() * mc_step);
	// printf("bef %d %d %d\n", mcu, kcu, ncu);
	mcu = clamp_val_int(mcu, mc_step, mc_max);
	kcu = clamp_val_int(kcu, kc_step, kc_max);
	ncu = clamp_val_int(ncu, nc_step, nc_max);
	// printf("%d %d %d\n\n", mcu, kcu, ncu);

	cand->m_c = mcu;
	cand->k_c = kcu;
	cand->n_c = ncu;
	cand->sch = curr->sch;
}



// Simulated annealing procedure to autotune for optimal (lowest runtime) sgemm parameters 
// (mc,kc,nc,sched) given an intial starting point of cake tiling and K-first scheduling
cache_dims_t* cake_autotune_sa(float* A, float* B, float* C, 
	int M, int K, int N, int p, int ntrials, int iters, int restarts) {
	
	double diff, original_runtime, curr_eval, best_eval, candidate_eval, metropolis;
	float t, temp = 1.0;
	cache_dims_t* best;
	cache_dims_t* cand = (cache_dims_t*) malloc(sizeof(cache_dims_t));
	cache_dims_t* curr = (cache_dims_t*) malloc(sizeof(cache_dims_t));

	cake_cntx_t* cake_cntx = cake_query_cntx();
	
	// initial starting point using cake tiling and K-first schedule
	best = cake_init_sa(M, K, N, p, cake_cntx);
    best_eval = cake_run(A, B, C, M, N, K, p, cake_cntx, ntrials,
						best->m_c, best->k_c, best->n_c, KMN);

    original_runtime = best_eval;
    int mco = best->m_c;
    int kco = best->k_c;
    int nco = best->n_c;

    curr_eval = best_eval;
	assign_cache_dims(curr, best);

	printf("iter %d, sch = %d, mc = %d, kc = %d, nc = %d\n", 
		-1, curr->sch, curr->m_c, curr->k_c, curr->n_c);

	// restart for annealing process
	for(int r = 0; r < restarts; r++) {

		// anneal for iters
	    for(int i = 0; i < iters; i++) {

	    	// printf("iter %d\n",i);
	    	// fflush(stdout);
	    	// select a valid, random (mc, kc, nc, sch) tuple by taking a random step with 
	    	// a Gaussian distribution where the mean is our current solution and 
	    	// the standard deviation is defined by the step_size
			// candidate = curr + randn(len(bounds)) * step_size
			get_candidate(M, K, N, p, cake_cntx, cand, curr);
			candidate_eval = cake_run(A, B, C, M, N, K, p, cake_cntx, ntrials, 
										cand->m_c, cand->k_c, cand->n_c, cand->sch);

	    	if(candidate_eval < best_eval) {
			    best_eval = candidate_eval;
			    assign_cache_dims(best, cand);
				printf("iter %d, sch = %d, mc = %d, kc = %d, nc = %d\n", 
					i, cand->sch, cand->m_c, cand->k_c, cand->n_c);
	    	}

	    	// diff = (candidate_eval - curr_eval) / curr_eval;
	    	diff = candidate_eval - curr_eval;
	    	// t = temp / ((float) (i + 1)); // temperature cooling
			t = temp - ((float) (i+1)) / ((float) iters); // linear cooling
	   		metropolis = exp(-diff / t); // we will update curr guess to candidate if this value is large 
	   		// printf("%f %f %f\n", diff, metropolis, t);
			// check if we should keep the candidate solution
			if((diff < 0) || (rand_gen() < metropolis)) {
				curr_eval = candidate_eval;
				assign_cache_dims(curr, cand);
			}
	    }
	}

    free(curr);
    free(cand);
    free(cake_cntx);
	printf("original param = %d,%d,%d\n new param = %d,%d,%d\n", mco,kco,nco, best->m_c, best->k_c, best->n_c);
	printf("original runtime = %f\n new runtime = %f\n", original_runtime, best_eval);
	double t1 = cake_reg_test(A, B, C, M, K, N, mco, kco, nco, p, ntrials);    
	double t2 = cake_reg_test(A, B, C, M, K, N, best->m_c, best->k_c, best->n_c, p, ntrials);    
	printf("original test = %f\n new test = %f\n", t1, t2);

    return best;
}




// int main( int argc, char** argv ) {
// 	 // run_tests();

//     if(argc < 3) {
//         printf("Enter M, K, and N\n");
//         exit(1);
//     }

// 	int M, K, N, p, ntrials, iters, restarts = 10;
// 	struct timespec start, end;
// 	double diff_t = 0.0;

// 	M = atoi(argv[1]);
// 	K = atoi(argv[2]);
// 	N = atoi(argv[3]);
// 	p = atoi(argv[4]);
//     ntrials = atoi(argv[5]);
//     iters = atoi(argv[6]);

// 	printf("M = %d, K = %d, N = %d, cores = %d\n", M,K,N,p);

// 	float* A = (float*) malloc(M * K * sizeof( float ));
// 	float* B = (float*) malloc(K * N * sizeof( float ));
// 	float* C = (float*) calloc(M * N , sizeof( float ));

// 	// initialize A and B
//     srand(time(NULL));
// 	rand_init(A, M, K);
// 	rand_init(B, K, N);

// 	clock_gettime(CLOCK_REALTIME, &start);

// 	cake_autotune_sa(A, B, C, M, K, N, p, ntrials, iters, restarts);    
// 	// cake_reg_test(A, B, C, M, K, N, 72,250,80, p, ntrials);    

//     clock_gettime(CLOCK_REALTIME, &end);
//     long seconds = end.tv_sec - start.tv_sec;
//     long nanoseconds = end.tv_nsec - start.tv_nsec;
//     diff_t = seconds + nanoseconds*1e-9;
// 	printf("autotune time: %f \n", diff_t ); 
	
// 	free(A);
// 	free(B);
// 	free(C);
	
// 	return 0;
// }





