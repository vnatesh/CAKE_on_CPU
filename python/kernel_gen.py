
# autogenerate sparse and dense outer product kernels of any dimension

class Haswell:
	def __init__(self, mr, nr):
		self.mr = mr
		self.nr = nr
		self.sparse = self.Sparse(self)
		self.dense = self.Dense(self)

	def gen_var_decls(self, m, n):
		ret = '''  
	int m_cnt, k_ind;
	__m256 a, '''
		for i in range(1, n/8):
			ret += "b%d, " % i
		ret += "b%d;" % (n/8)
		ret += '''
	__m256 c[%d*%d];
	''' % (m,n/8)
		return ret

	def gen_C_load(self, m, n):
		C_load = '''
	// load tile of C into AVX2 registers
	c[0]  = _mm256_loadu_ps(C);'''
		for i in range(1, m*n / 8):
			C_load += '''
	c[%d]  = _mm256_loadu_ps(C + %d);''' % (i,i*8)
		return C_load

	def gen_C_store(self, m, n):
		C_store = '''
	_mm256_storeu_ps(C, c[0]);'''
		for i in range(1, m*n / 8):
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
			nlanes = n/8
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

		B += n;
				'''
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
			ret += self.gen_inner_kernel(m, n)
			for i in range(1,4):
				ret += self.gen_inner_kernel(m, n) + '\n\n'
			ret += '''
	}
			'''
			return ret


		def gen_kernel(self):
			m = self.mr
			n = self.nr
			func = self.gen_func_def( m, n)
			var_decls = self.haswell.gen_var_decls(m, n)
			C_load = self.haswell.gen_C_load( m, n)
			outer_prod = self.gen_outer_prod_loop(m, n)
			leftover = self.gen_leftover_k( m, n)
			C_store = self.haswell.gen_C_store( m, n)
			return func + var_decls + C_load + outer_prod + leftover + C_store



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
			nlanes = n/8
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

		def gen_kernel(self):
			m = self.mr
			n = self.nr
			func = self.gen_func_def( m, n)
			var_decls = self.haswell.gen_var_decls(m, n)
			C_load = self.haswell.gen_C_load( m, n)
			outer_prod = self.gen_outer_prod_loop(m, n)
			leftover = self.gen_leftover_k( m, n)
			C_store = self.haswell.gen_C_store( m, n)
			return func + var_decls + C_load + outer_prod + leftover + C_store

# from kernel_gen import *
# a = Haswell(6,16)
# b = a.dense.gen_kernel()
# print(b)
# #

# print(gen_sparse_kernel(6,16))
