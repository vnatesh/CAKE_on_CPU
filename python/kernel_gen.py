import sys
# autogenerate sparse and dense outer product kernels of any dimension for Haswell and Armv8



class Haswell:
	def __init__(self, mr, nr):
		self.mr = mr
		self.nr = nr
		self.sparse = self.Sparse(self)
		self.dense = self.Dense(self)

	def gen_var_decls(self, m, n):
		ret = '''  
	__m256 a, '''
		for i in range(1, n//8):
			ret += "b%d, " % i
		ret += "b%d;" % (n//8)
		ret += '''
	__m256 c[%d*%d];
	''' % (m,n//8)
		return ret

	def gen_C_load(self, m, n):
		C_load = '''
	// load tile of C into AVX2 registers
	c[0]  = _mm256_loadu_ps(C);'''
		for i in range(1, m*n // 8):
			C_load += '''
	c[%d]  = _mm256_loadu_ps(C + %d);''' % (i,i*8)
		return C_load

	def gen_C_store(self, m, n):
		C_store = '''
	_mm256_storeu_ps(C, c[0]);'''
		for i in range(1, m*n // 8):
			C_store += '''
	_mm256_storeu_ps((C + %d), c[%d]);''' % (i*8,i)
		C_store += '''
}'''
		return C_store


	class Dense:
		def __init__(self, haswell):
			self.mr = haswell.mr
			self.nr = haswell.nr
			self.haswell = haswell
			
		def gen_func_def(self, m, n):
			return '''
void cake_sgemm_haswell_%dx%d(float* A, float* B, float* C, int m, int n, int k) {
			''' % (m,n)

		def gen_inner_kernel(self, m, n):
			nlanes = n//8
			ret = '''
		b1 = _mm256_load_ps(B);'''
			for i in range(1, nlanes):
				ret += '''
		b%d = _mm256_load_ps(B + %d);''' % (i+1, i*8)
			for i in range(m):
				ret += '''

		a = _mm256_broadcast_ss(A++);'''
				for j in range(nlanes):
					ret += '''
		c[%d] =  _mm256_fmadd_ps(a, b%d, c[%d]);''' % \
				(i*nlanes + j, j+1, i*nlanes + j)
			ret += '''

		B += n;'''
			return ret


# print(gen_inner_kernel(6,16))
				
		def gen_leftover_k(self, m, n):
			ret = '''
	for(int kk = 0; kk < rem; kk++) { 
			'''
			ret += self.gen_inner_kernel(m, n)
			ret += '''
	}
			'''
			return ret

		def gen_outer_prod_loop(self, m, n):
			ret = '''

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

			'''
			ret += self.gen_inner_kernel(m, n) + '\n\n'
			for i in range(1,4):
				ret += self.gen_inner_kernel(m, n) + '\n\n'
			ret += '''
	}
	'''
			return ret



	class Sparse:
		def __init__(self, haswell):
			self.mr = haswell.mr
			self.nr = haswell.nr
			self.haswell = haswell

		def gen_func_def(self, m, n):
			return '''
void cake_sp_sgemm_haswell_%dx%d(float* A, float* B, float* C, int m, int n, int k, 
									char* nnz_outer, int* k_inds, char* loc_m) {
			''' % (m,n)

		def gen_inner_kernel(self, m, n):
			nlanes = n//8
			ret = '''
		b1 = _mm256_load_ps(B + k_ind);'''
			for i in range(1, nlanes):
				ret += '''
		b%d = _mm256_load_ps(B + k_ind + %d);''' % (i+1, i*8)
			ret += '''
			
		for(int j = 0; j < m_cnt; j++) {
			a = _mm256_broadcast_ss(A++);
			c[*loc_m * %d]     =  _mm256_fmadd_ps(a, b1, c[*loc_m * %d]);''' % (nlanes, nlanes)
			for i in range(1, nlanes):
				ret += '''
			c[*loc_m * %d + %d] =  _mm256_fmadd_ps(a, b%d, c[*loc_m * %d + %d]);''' % (nlanes, i, i+1, nlanes, i)
			ret += '''
			loc_m++;
		}
				'''
			return ret
				
		def gen_leftover_k(self, m, n):
			ret = '''
	for(int kk = 0; kk < rem; kk++) { 

		m_cnt = nnz_outer[k + kk];
		k_ind = n*k_inds[k + kk];
			'''
			ret += self.gen_inner_kernel(m, n)
			ret += '''
	}
			'''
			return ret

		def gen_outer_prod_loop(self, m, n):
			ret = '''

	int rem = k % 4;
	k -= rem;

	// float* B_ptr = &B[0];

	for(int kk = 0; kk < k; kk += 4) { 

		m_cnt = nnz_outer[kk];

		// skip columns with 0 nonzeros
		if(!m_cnt) {
			break;
		}

		k_ind = n*k_inds[kk];
			'''
			ret += self.gen_inner_kernel(m, n)
			for i in range(1,4):
				ret += '''
		m_cnt = nnz_outer[kk+%d];
		k_ind = n*k_inds[kk+%d];
				''' % (i,i)
				ret += self.gen_inner_kernel(m, n)
			ret += '''
	}
			'''
			return ret




class Armv8:
	def __init__(self, mr, nr):
		self.mr = mr
		self.nr = nr
		self.sparse = self.Sparse(self)
		self.dense = self.Dense(self)

	def gen_var_decls(self, m, n):
		ret = '''  
	float32x4_t a, '''
		for i in range(1, n//4):
			ret += "b%d, " % i
		ret += "b%d;" % (n//4)
		ret += '''
	float32x4_t c[%d*%d];
	''' % (m,n//4)
		return ret

	def gen_C_load(self, m, n):
		C_load = '''
	// load tile of C into arm neon SIMD registers
	c[0]  = vld1q_f32(C);'''
		for i in range(1, m*n // 4):
			C_load += '''
	c[%d]  = vld1q_f32(C + %d);''' % (i,i*4)
		return C_load

	def gen_C_store(self, m, n):
		C_store = '''
	vst1q_f32(C, c[0]);'''
		for i in range(1, m*n // 4):
			C_store += '''
	vst1q_f32(C + %d, c[%d]);''' % (i*4,i)
		C_store += '''
}'''
		return C_store


	class Dense:
		def __init__(self, armv8):
			self.mr = armv8.mr
			self.nr = armv8.nr
			self.armv8 = armv8
			
		def gen_func_def(self, m, n):
			return '''
void cake_sgemm_armv8_%dx%d(float* A, float* B, float* C, int m, int n, int k) {
			''' % (m,n)

		def gen_inner_kernel(self, m, n):
			nlanes = n//4
			ret = '''
		b1 = vld1q_f32(B);'''
			for i in range(1, nlanes):
				ret += '''
		b%d = vld1q_f32(B + %d);''' % (i+1, i*4)
			for i in range(m):
				ret += '''

		a = vld1q_dup_f32(A++);'''
				for j in range(nlanes):
					ret += '''
		c[%d] =  vfmaq_f32(c[%d], b%d, a);''' % \
				(i*nlanes + j, i*nlanes + j, j+1)
			ret += '''

		B += n;'''
			return ret


# print(gen_inner_kernel(6,16))
				
		def gen_leftover_k(self, m, n):
			ret = '''
	for(int kk = 0; kk < rem; kk++) { 
			'''
			ret += self.gen_inner_kernel(m, n)
			ret += '''
			}
			'''
			return ret

		def gen_outer_prod_loop(self, m, n):
			ret = '''

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

			'''
			ret += self.gen_inner_kernel(m, n) + '\n\n'
			for i in range(1,4):
				ret += self.gen_inner_kernel(m, n) + '\n\n'
			ret += '''
	}
			'''
			return ret



	class Sparse:
		def __init__(self, armv8):
			self.mr = armv8.mr
			self.nr = armv8.nr
			self.armv8 = armv8

		def gen_func_def(self, m, n):
			return '''
void cake_sp_sgemm_armv8_%dx%d(float* A, float* B, float* C, int m, int n, int k, 
									char* nnz_outer, int* k_inds, char* loc_m) {
			''' % (m,n)

		def gen_inner_kernel(self, m, n):
			nlanes = n//4
			ret = '''
		b1 = vld1q_f32(B + k_ind);'''
			for i in range(1, nlanes):
				ret += '''
		b%d = vld1q_f32(B + k_ind + %d);''' % (i+1, i*4)
			ret += '''
			
		for(int j = 0; j < m_cnt; j++) {
			a = vld1q_dup_f32(A++);
			c[*loc_m * %d]     =  vfmaq_f32(c[*loc_m * %d], b1, a);''' % (nlanes, nlanes)
			for i in range(1, nlanes):
				ret += '''
			c[*loc_m * %d + %d] =  vfmaq_f32(c[*loc_m * %d + %d], b%d, a);''' % (nlanes, i, nlanes, i, i+1)
			ret += '''
			loc_m++;
		}
				'''
			return ret
				
		def gen_leftover_k(self, m, n):
			ret = '''
	for(int kk = 0; kk < rem; kk++) { 

		m_cnt = nnz_outer[k + kk];
		k_ind = n*k_inds[k + kk];
			'''
			ret += self.gen_inner_kernel(m, n)
			ret += '''
	}
			'''
			return ret

		def gen_outer_prod_loop(self, m, n):
			ret = '''

	int rem = k % 4;
	k -= rem;

	// float* B_ptr = &B[0];

	for(int kk = 0; kk < k; kk += 4) { 

		m_cnt = nnz_outer[kk];

		// skip columns with 0 nonzeros
		if(!m_cnt) {
			break;
		}

		k_ind = n*k_inds[kk];
			'''
			ret += self.gen_inner_kernel(m, n)
			for i in range(1,4):
				ret += '''
		m_cnt = nnz_outer[kk+%d];
		k_ind = n*k_inds[kk+%d];
				''' % (i,i)
				ret += self.gen_inner_kernel(m, n)
			ret += '''
	}
			'''
			return ret




def gen_kernel(arch, mr, nr, op):
	arch = arch(mr, nr)
	m = arch.mr
	n = arch.nr
	if op == 'sparse':
		op = arch.sparse
		sp = '''
int m_cnt, k_ind;'''
	else:
		op = arch.dense
		sp = ''
	func = op.gen_func_def(m, n)
	var_decls = arch.gen_var_decls(m, n)
	C_load = arch.gen_C_load(m, n)
	outer_prod = op.gen_outer_prod_loop(m, n)
	leftover = op.gen_leftover_k(m, n)
	C_store = arch.gen_C_store(m, n)
	return func + sp + var_decls + C_load + outer_prod + leftover + C_store


def gen_kernel_headers(arch):
	fact = 12 if arch == 'armv8' else 16
	ret = '''
#include "common.h"

typedef void cake_sp_sgemm_%s(float* A, float* B, float* C, int m, int n, int k, 
									char* nnz_outer, int* k_inds, char* loc_m);
typedef void cake_sgemm_%s(float* A, float* B, float* C, int m, int n, int k);
''' % (arch, arch)
	for i in range(1,11):
		for j in range(1,7):
			ret += '''
void cake_sp_sgemm_%s_%dx%d(float* A, float* B, float* C, int m, int n, int k, 
									char* nnz_outer, int* k_inds, char* loc_m);
void cake_sgemm_%s_%dx%d(float* A, float* B, float* C, int m, int n, int k);
									''' % (arch, i*2,j*fact, arch, i*2,j*fact)	
	sparse_arr = []
	dense_arr = []
	for i in range(1,11):
		sparse_arr.append('''
	{'''+','.join(['cake_sp_sgemm_%s_%dx%d' % (arch, i*2,j*fact) for j in range(1,7)]) + '}')
		dense_arr.append('''
	{'''+','.join(['cake_sgemm_%s_%dx%d' % (arch, i*2,j*fact) for j in range(1,7)]) + '}')	
	ret += '''
static cake_sp_sgemm_%s* kernel_map_sp[10][6] = 
{
	''' % arch
	ret += ','.join(sparse_arr) + '''
};'''
	ret += '''
static cake_sgemm_%s* kernel_map[10][6] = 
{
	''' % arch
	ret += ','.join(dense_arr) + '''
};'''
	ret += '''
// kernel helper functions
inline void cake_sgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, cake_cntx_t* cake_cntx) {


#ifdef USE_BLIS 
		// Set the scalars to use during each GEMM kernel call.
	float alpha_blis = 1.0;
	float beta_blis  = 1.0;
	// inc_t rsc, csc;
	// rsc = n_r; csc = 1;
    // rsc = 1; csc = m_r;
	auxinfo_t def_data;

	bli_sgemm_haswell_asm_6x16(k_c_t, &alpha_blis, A_p, B_p, &beta_blis, C_p, 
		(inc_t) n_r, (inc_t) 1, &def_data, (cntx_t*) cake_cntx->blis_cntx);

#elif USE_CAKE_HASWELL
	cake_sgemm_haswell_6x16(A_p, B_p, C_p, m_r, n_r, k_c_t);
#elif USE_CAKE_ARMV8
	cake_sgemm_armv8_8x12(A_p, B_p, C_p, m_r, n_r, k_c_t);
#endif

}


inline void cake_spgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, char* nnz_outer, int* k_inds, char* loc_m) {

#ifdef USE_CAKE_HASWELL
	cake_sp_sgemm_haswell_6x16(A_p, B_p, C_p, m_r, n_r, k_c_t, nnz_outer, k_inds, loc_m);
#elif USE_CAKE_ARMV8
	cake_sp_sgemm_armv8_8x12(A_p, B_p, C_p, m_r, n_r, k_c_t, nnz_outer, k_inds, loc_m);
#endif

}
	'''
	f = open("kernels.h", 'w')
	f.write(ret)





def gen_all_kernels(arch):
	if arch == 'armv8':
		arch_class = Armv8
		fact = 12
	else:
		arch_class = Haswell
		fact = 16
	ret1 = ret2 = '''
#include "cake.h"
	'''
	for i in range(1,11):
		for j in range(1,7):
			ret1 += gen_kernel(arch_class, i*2,j*fact, 'sparse')
			ret2 += gen_kernel(arch_class, i*2,j*fact, 'dense')
	f1 = open("sparse.cpp", 'w')
	f1.write(ret1)
	f2 = open("dense.cpp", 'w')
	f2.write(ret2)


if __name__ == '__main__':
	gen_kernel_headers(sys.argv[1])
	gen_all_kernels(sys.argv[1])


# from kernel_gen import *
# a = gen_kernel(Haswell, 6, 16, 'dense')
# a = Haswell(6,16)
# b = a.dense.gen_kernel()
# print(b)
# #



# print(gen_sparse_kernel(6,16))
