#include "common.h"




// retrieve system parameter information
cake_cntx_t* cake_query_cntx();

cake_cntx_t* cake_query_cntx_torch(int L2, int L3);

int get_cache_size(int level);

int lcm(int n1, int n2);

int get_num_physical_cores();



// schedule derivation and block sizing 
enum sched derive_schedule(int M, int N, int K, int p, 
					int mc_ret, cake_cntx_t* cake_cntx);

cache_dims_t* get_cache_dims(int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], float sparsity = 0);

void init_block_dims(int M, int N, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], float sparsity = 0);


