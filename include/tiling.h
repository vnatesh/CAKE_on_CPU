
#include "common.h"



#define cake_max(x,y) (((x) >= (y)) ? (x) : (y))
#define cake_min(x,y) (((x) <= (y)) ? (x) : (y))


double clamp_val(double d, double min, double max);



// retrieve system parameter information
cake_cntx_t* cake_query_cntx();

cake_cntx_t* cake_query_cntx_torch(int L2, int L3);

void update_mr_nr(cake_cntx_t* cake_cntx, int m_r, int n_r);

int get_cache_size(int level);

int lcm(int n1, int n2);

int get_num_physical_cores();



// schedule derivation and block sizing 
enum sched derive_schedule(int M, int N, int K, int p, 
					int mc_ret, cake_cntx_t* cake_cntx);


// default type size = 4 bytes for float
cache_dims_t* get_cache_dims(int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float type_size = 4, int mcu = 0, int kcu = 0, int ncu = 0);

void init_block_dims(int M, int N, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float type_size = 4, int mcu = 0, int kcu = 0, int ncu = 0);

void init_block_dims_2d(int M, int N, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	 float type_size = 4, int mcu = 0, int kcu = 0, int ncu = 0);

void init_block_dims_2d_small(int M, int N, int K, int p, blk_dims_t* x, 
	cake_cntx_t* cake_cntx, enum sched sch, char* argv[], 
	float type_size = 4);

int grid_dims_2d(int M, int N, int K, int p, int ncores);




