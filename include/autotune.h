#include <stdio.h>
#include <tuple>
#include <unordered_set>
#include <iostream>


using std::hash;
using std::tuple;
using std::make_tuple;
using std::unordered_set;



// hash function for tuple containing kernel params (mc,kc,nc,mr,nr,etc)
namespace hash_tuple{

    template <typename TT>
    struct hash
    {
        size_t
        operator()(TT const& tt) const
        {                                              
            return std::hash<TT>()(tt);                                 
        }                                              
    };
}

namespace hash_tuple{
    namespace
    {
        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v)
        {
            seed ^= hash_tuple::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
    }
}


namespace hash_tuple{

    namespace
    {
        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, std::get<Index>(tuple));
          }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0>
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            hash_combine(seed, std::get<0>(tuple));
          }
        };
    }

    template <typename ... TT>
    struct hash<std::tuple<TT...>> 
    {
        size_t
        operator()(std::tuple<TT...> const& tt) const
        {                                              
            size_t seed = 0;                             
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);    
            return seed;                                 
        }                                              
    }; 
}


// unordered_set<tuple<int, int, int>, hash_tuple::hash<tuple<int, int, int>>> visited_params;


void assign_cache_dims(cache_dims_t* a, cache_dims_t* b);
int clamp_val_int(int d, int min, int max);
int get_random_step(int step);
void get_candidate(int M, int K, int N, int pm, int pn, cake_cntx_t* cake_cntx,
    cache_dims_t* cand, cache_dims_t* curr);
