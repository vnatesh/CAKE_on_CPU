#include "cake.h"


#define kc_step 20
#define mc_step 30
#define nc_step 128


unordered_set<tuple<int, int, int>, hash_tuple::hash<tuple<int, int, int>>> visited_params;



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
void get_candidate(int M, int K, int N, int pm, int pn, cake_cntx_t* cake_cntx,
    cache_dims_t* cand, cache_dims_t* curr) {

    int mcu, kcu, ncu;
    int mr = cake_cntx->mr, nr = cake_cntx->nr;

    int mc_max = (M / pm) + (mr - ((M / pm) % mr));
    // int mc_max = M + (mr - (M % mr));
    int kc_max = K;
    int nc_max = (N / pn) + (nr - ((N / pn) % nr));


    int max_cand =  ((M % mc_step) ? (M / mc_step) + 1 : (M / mc_step)) * 
                    ((K % kc_step) ? (K / kc_step) + 1 : (K / kc_step)) * 
                    ((N % nc_step) ? (N / nc_step) + 1 : (N / nc_step));
    // printf("bef1 %d %d %d\n", curr->m_c, curr->k_c, curr->n_c);
                    printf("heyyy %d\n", max_cand);
    while(visited_params.size() < max_cand) {

        printf("%ld ", visited_params.size());

        mcu = curr->m_c + get_random_step(mc_step);
        kcu = curr->k_c + get_random_step(kc_step);
        ncu = curr->n_c + get_random_step(nc_step);

        // printf("%d\n", curr->m_c +   normalRandom() * mc_step);
        // printf("bef %d %d %d\n", mcu, kcu, ncu);
        mcu = clamp_val_int(mcu, mr, mc_max);
        kcu = clamp_val_int(kcu, kc_step, kc_max);
        ncu = clamp_val_int(ncu, nr, nc_max);

        tuple<int, int, int> param = make_tuple(mcu, kcu, ncu);

        if(visited_params.count(param)) {
            printf("s ");
            continue;
        } else {
            visited_params.insert(param);
            cand->m_c = mcu;
            cand->k_c = kcu;
            cand->n_c = ncu;
            cand->sch = curr->sch;
            printf("%d %d %d\n\n", mcu, kcu, ncu);
            break;
        }
    }
}