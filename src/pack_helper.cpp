#include "cake.h"



size_t cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	int mr_rem, M_padded;

	switch(sch) {

		case KMN: {
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
			M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);
			break;
		}

		case MKN: {
			mr_rem = (int) ceil( ((double) (M % x->m_c)) / cake_cntx->mr);
			M_padded = (M / x->m_c)*x->m_c + mr_rem*cake_cntx->mr;
			break;
		}

		case NKM: {
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
			M_padded = (cake_cntx->mr*mr_rem + (M /(p*x->m_c))*p*x->m_c);
			break;
		}

		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}

	return (M_padded * K) * sizeof(float);
}



size_t cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {
	
	int nr_rem = (int) ceil( ((double) (N % x->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N % x->n_c)) + n_c1;

	return (K * N_padded) * sizeof(float);
}



size_t cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	int mr_rem, M_padded;

	switch(sch) {

		case KMN: {
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr) ;
			M_padded = (cake_cntx->mr*mr_rem + (M / (p*x->m_c))*p*x->m_c);
			break;
		}

		case MKN: {
			mr_rem = (int) ceil( ((double) (M % x->m_c)) / cake_cntx->mr);
			M_padded = (M / x->m_c)*x->m_c + mr_rem*cake_cntx->mr;
			break;
		}

		case NKM: {			
			mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / cake_cntx->mr);
			M_padded = (cake_cntx->mr*mr_rem + (M / (p*x->m_c))*p*x->m_c);
			break;
		}

		default: {
			printf("unknown schedule\n");
			exit(1);
		}
	}


	int nr_rem = (int) ceil( ((double) (N % x->n_c) / cake_cntx->nr)) ;
	int n_c1 = nr_rem * cake_cntx->nr;
	int N_padded = (N - (N % x->n_c)) + n_c1;

	return (M_padded * N_padded) * sizeof(float);
}




void pack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			pack_C_single_buf_k_first(C, C_p, M, N, p, x, cake_cntx);
			break;
		}
		case MKN: {
			pack_C_single_buf_m_first(C, C_p, M, N, p, x, cake_cntx);
			break;
		}
		case NKM: {
			pack_C_single_buf_n_first(C, C_p, M, N, p, x, cake_cntx);
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}

}


void pack_B(float* B, float* B_p, int K, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			pack_B_k_first(B, B_p, K, N, x, cake_cntx);
			break;
		}
		case MKN: {
			pack_B_m_first(B, B_p, K, N, p, x, cake_cntx);
			break;
		}
		case NKM: {
			pack_B_n_first(B, B_p, K, N, p, x, cake_cntx);
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}



double pack_A(float* A, float* A_p, int M, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			return pack_A_single_buf_k_first(A, A_p, M, K, p, x, cake_cntx);
		}
		case MKN: {
			return pack_A_single_buf_m_first(A, A_p, M, K, p, x, cake_cntx); 
		}
		case NKM: {
			return pack_A_single_buf_n_first(A, A_p, M, K, p, x, cake_cntx);
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}





void pack_A_sp(float* A, float* A_p, int M, int K, int p, 
   sp_pack_t* sp_pack, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			return pack_A_sp_k_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
		}
		case MKN: {
			return pack_A_sp_m_first(A, A_p, M, K, p, sp_pack, x, cake_cntx); 
		}
		case NKM: {
			return pack_A_sp_n_first(A, A_p, M, K, p, sp_pack, x, cake_cntx);
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}



void unpack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) {

	switch(sch) {
		case KMN: {
			unpack_C_single_buf_k_first(C, C_p, M, N, p, x, cake_cntx); 
			break;
		}
		case MKN: {
			unpack_C_single_buf_m_first(C, C_p, M, N, p, x, cake_cntx); 
			break;
		}
		case NKM: {
			unpack_C_single_buf_n_first(C, C_p, M, N, p, x, cake_cntx); 
			break;
		}
		default: {
			printf("unknown schedule\n");
			exit(1);
		}	
	}
}




sp_pack_t* malloc_sp_pack(int M, int K, int nz, blk_dims_t* x, cake_cntx_t* cake_cntx) {

	sp_pack_t* sp_pack = (sp_pack_t*) malloc(sizeof(sp_pack_t));
	sp_pack->A_sp_p = (float*) calloc(nz, sizeof(float)); // storing only nonzeros of A                                        
	sp_pack->loc_m = (char*) calloc(nz , sizeof(char)); // array for storing M dim C writeback location for each nnz in A
	           // each value ranges from 0 to mr-1
	sp_pack->nnz_outer = (char*) calloc(nz , sizeof(char)); // storing number of nonzeros 
	                                                 // in each outer prod col of A
	sp_pack->k_inds = (int*) calloc(nz , sizeof(int)); // storing kc_ind 
	                                                 // of each outer prod col of A
	sp_pack->nnz_tiles = (int*) calloc((x->M_padded / cake_cntx->mr)*x->Kb + 1 , sizeof(int)); 
	sp_pack->num_col_tile = (int*) calloc((x->M_padded / cake_cntx->mr)*x->Kb + 1, sizeof(int)); 
	sp_pack->M = M;
	sp_pack->K = K;
	sp_pack->nnz = nz;

	return sp_pack;
}



void free_sp_pack(sp_pack_t* x) {
	free(x->loc_m);
	free(x->nnz_outer);
	free(x->k_inds);
	free(x->A_sp_p);
	free(x->nnz_tiles);
	free(x->num_col_tile);
}



// write matrix packed in rosko format to binary file
// M,K,nnz,nnz_cols,ntiles,loc_m,nnz_outer,k_inds,A_sp_p,nnz_tiles,num_col_tile
void sp_pack_to_file(sp_pack_t* sp_pack, char* fname) {

	FILE *fptr = fopen(fname, "wb");

	int tmp[5];
	tmp[0] = sp_pack->M;
	tmp[1] = sp_pack->K;
	tmp[2] = sp_pack->nnz;
	tmp[3] = sp_pack->nnz_cols;
	tmp[4] = sp_pack->ntiles;

	fwrite(&tmp, sizeof(int), 5, fptr);
	fwrite(sp_pack->loc_m, sizeof(char), sp_pack->nnz, fptr);
	fwrite(sp_pack->nnz_outer, sizeof(char), sp_pack->nnz_cols, fptr);
	fwrite(sp_pack->k_inds, sizeof(int), sp_pack->nnz_cols, fptr);
	fwrite(sp_pack->A_sp_p, sizeof(float), sp_pack->nnz, fptr);
	fwrite(sp_pack->nnz_tiles, sizeof(int), sp_pack->ntiles, fptr);
	fwrite(sp_pack->num_col_tile, sizeof(int), sp_pack->ntiles, fptr);

	fclose(fptr);
}



void free_csr(csr_t* x) {
	free(x->rowptr);
	free(x->colind);
	free(x->vals);
	free(x);
}


int mat_to_csr_file(float* A, int M, int K, char* fname) {

	float* vals = (float*) malloc(M * K * sizeof(float));
	int* colind = (int*) malloc(M * K * sizeof(int));
	int* rowptr = (int*) malloc((M+1) * sizeof(int));
	rowptr[0] = 0;

	FILE *fptr = fopen(fname, "wb");
	int nz = 0;

	for(int i = 0; i < M; i++) {
		for(int j = 0; j < K; j++) {
			float tmp = A[i*K + j];
			if(tmp != 0) {
				vals[nz] = tmp;
				colind[nz] = j;
				nz++;
			}
		}

		rowptr[i+1] = nz;
	}

	int tmp[3];
	tmp[0] = M;
	tmp[1] = K;
	tmp[2] = nz;

	fwrite(&tmp, sizeof(int), 3, fptr);
	fwrite(rowptr, sizeof(int), (M + 1), fptr);
	fwrite(colind, sizeof(int), nz, fptr);
	fwrite(vals, sizeof(float), nz, fptr);

	fclose(fptr);
	free(rowptr); free(vals); free(colind);
	return nz;
}



// read in CSR matrix from file
// M,K,nnz,rowptr,colind,vals
csr_t* file_to_csr(char* fname) {

	int M, K, nz;

	FILE *fptr = fopen(fname, "rb");
	if (fptr == NULL) {
	   perror("fopen");
	   exit(EXIT_FAILURE);
	}

	int tmp[3];
	fread(&tmp, sizeof(int), 3, fptr);

	M = tmp[0];
	K = tmp[1];
	nz = tmp[2];

	int* rowptr = (int*) malloc((M + 1) * sizeof(int));
	int* colind = (int*) malloc(nz * sizeof(int));
	float* vals = (float*) malloc(nz * sizeof(float));

	fread(rowptr, sizeof(int), (M + 1), fptr);
	fread(colind, sizeof(int), nz, fptr);
	fread(vals, sizeof(float), nz, fptr);

	// printf("M = %d K = %d nz = %d\n", M, K, nz);

   	fclose(fptr);

	csr_t* csr_ret = (csr_t*) malloc(sizeof(csr_t));
	csr_ret->rowptr = rowptr;
	csr_ret->colind = colind;
	csr_ret->vals = vals;
	csr_ret->M = M;
	csr_ret->K = K;

   	return csr_ret;
}



void csr_to_mat(float* A, int M, int K, int* rowptr, float* vals, int* colind) {

	int ks, ind = 0;

	for(int i = 0; i < M; i++) {
		ks = rowptr[i+1] - rowptr[i];
		for(int j = 0; j < ks; j++) {
			A[i*K + colind[ind]] = vals[ind];
			ind++;
		}
	}
}


void test_csr_convert(int M, int K, float sparsity) {

	char fname[50];
	snprintf(fname, sizeof(fname), "convert_test");
	float* A = (float*) malloc(M * K * sizeof( float ));
	float* A_check = (float*) malloc(M * K * sizeof(float));

    srand(time(NULL));
	rand_sparse(A, M, K, ((float) sparsity) / 100.0);

	int nz = mat_to_csr_file(A, M, K, fname);
	csr_t* csr = file_to_csr(fname);
	csr_to_mat(A_check, M, K, csr->rowptr, csr->vals, csr->colind);
	mat_equals(A, A_check, M, K);
	free(A); free(A_check);

	// for(int i = 0; i < M+1; i++) {
	// 	printf("%d ", csr->rowptr[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < nz; i++) {
	// 	printf("%d ", csr->colind[i]);
	// }
	// printf("\n");


	// for(int i = 0; i < nz; i++) {
	// 	printf("%f ", csr->vals[i]);
	// }
	// printf("\n");
}


