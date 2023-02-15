#include "cake.h"




// void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

//    int ind_ob = 0;

//    if(pad) {
//       for(int m3 = 0; m3 < m_c; m3 += m_r) {
//          for(int i = 0; i < k_c; i++) {
//             for(int j = 0; j < m_r; j++) {
//                if((m1 + m2 + m3 + j) <  M) {
//                   A_p[ind_ob] = A[m3*K + i + j*K];
//                }

//                ind_ob++;
//             }
//          }
//       }     
//    } 

//    else {
//       for(int m3 = 0; m3 < m_c; m3 += m_r) {
//          for(int i = 0; i < k_c; i++) {
//             for(int j = 0; j < m_r; j++) {
//                A_p[ind_ob] = A[m3*K + i + j*K];
//                ind_ob++;
//             }
//          }
//       }     
//    }
// }





void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int ind_ob = 0;

   for(int m3 = 0; m3 < m_c; m3 += m_r) {
      for(int i = 0; i < k_c; i++) {
         for(int j = 0; j < m_r; j++) {

            if((m1 + m2 + m3 + j) >=  M) {
               A_p[ind_ob] = 0.0;
            } else {
               // printf("PAD IND %d\n", m3*K + i + j*K);

               A_p[ind_ob] = A[m3*K + i + j*K];
            }

            ind_ob++;
            // printf("ind_ob %d\n", ind_ob);
         }
      }
   }     

   // if(pad) {
   // printf("PAD m1 %d m2 %d mc %d kc %d pad %d\n", m1, m2, m_c, k_c, pad);

   //    for(int m3 = 0; m3 < m_c; m3 += m_r) {
   //       for(int i = 0; i < k_c; i++) {
   //          for(int j = 0; j < m_r; j++) {

   //             if((m1 + m2 + m3 + j) >=  M) {
   //                A_p[ind_ob] = 0.0;
   //             } else {
   //                printf("PAD IND %d\n", m3*K + i + j*K);

   //                A_p[ind_ob] = A[m3*K + i + j*K];
   //             }

   //             ind_ob++;
   //             // printf("ind_ob %d\n", ind_ob);
   //          }
   //       }
   //    }     
   // } 

   // else {
   //    printf("m1 %d m2 %d mc %d kc %d pad %d\n", m1, m2, m_c, k_c, pad);

   //    for(int m3 = 0; m3 < m_c; m3 += m_r) {
   //       for(int i = 0; i < k_c; i++) {
   //          for(int j = 0; j < m_r; j++) {
   //                printf("IND %d\n", m3*K + i + j*K);

   //             A_p[ind_ob] = A[m3*K + i + j*K];

   //             ind_ob++;
   //          }
   //       }
   //    }     
   // }
}





void pack_ob_B_single_buf(float* B, float* B_p, int K, int N, int n1,
            int k_c, int n_c, int n_r, bool pad_n) {

   int ind_ob = 0;

   if(pad_n) {
      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) >=  N) {
                     B_p[ind_ob] = 0.0;
               } else {
                  // B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = B[n1 + k1*N + n2 + i*N + j];
                  B_p[ind_ob] = B[n2 + i*N + j];
               }
               ind_ob++;
            }
         }
      }
   }
   
   else {
      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               B_p[ind_ob] = B[n2 + i*N + j];
               ind_ob++;
            }

            // _mm256_store_ps (&B_p[ind_ob], _mm256_load_ps(&B[n2 + i*N]));
            // _mm256_store_ps (&B_p[ind_ob + 8], _mm256_load_ps(&B[n2 + i*N + 8]));
            // ind_ob += n_r;
         }
      }
   }
}



// B row stored, B_p also row stored (k_c rows with n_r contiguous elements each)
void pack_B_nr_x_kc(float* B, float* B_p, int N, int k_c, int n_r) {

   int ind_ob = 0;

   for(int i = 0; i < k_c; i++) {
      for(int j = 0; j < n_r; j++) {
         B_p[ind_ob] = B[i*N + j];
         ind_ob++;
      }
   }
}


// A row stored, A_p contains k_c cols of mr contiguous elements
void pack_A_mr_x_kc(float* A, float* A_p, int K, int k_c, int m_r) {

   int ind_ob = 0;

   for(int i = 0; i < k_c; i++) {
      for(int j = 0; j < m_r; j++) {
         A_p[ind_ob] = A[i + j*K];
         ind_ob++;
      }
   }       
}



void pack_ob_B_parallel(float* B, float* B_p, int K, int N, int n1,
            int k_c, int n_c, int n_r, bool pad_n) {


   int ind_ob, n2;

   if(pad_n) {
      #pragma omp parallel for private(n2, ind_ob)
      for(n2 = 0; n2 < n_c; n2 += n_r) {
         ind_ob = 0;
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) >=  N) {
                  B_p[n2*k_c + ind_ob] = 0.0;
               } else {
            // B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = B[n1 + k1*N + n2 + i*N + j];
                  B_p[n2*k_c + ind_ob] = B[n2 + i*N + j];
               }
               ind_ob++;
            }
         }
      }
   }

   else {
      #pragma omp parallel for private(n2, ind_ob)
      for(n2 = 0; n2 < n_c; n2 += n_r) {
         ind_ob = 0;
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < n_r; j++) {
               B_p[n2*k_c + ind_ob] = B[n2 + i*N + j];
               ind_ob++;
            }

   // _mm256_store_ps (&B_p[ind_ob], _mm256_load_ps(&B[n2 + i*N]));
   // _mm256_store_ps (&B_p[ind_ob + 8], _mm256_load_ps(&B[n2 + i*N + 8]));
   // ind_ob += n_r;
         }
      }
   }
}

// void pack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
//             int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n) {

//    int ind_ob = 0;

//    if(pad_m || pad_n) {

//       for(int n2 = 0; n2 < n_c; n2 += n_r) {
//          for(int m3 = 0; m3 < m_c; m3 += m_r) {
//             for(int i = 0; i < m_r; i++) {
//                for(int j = 0; j < n_r; j++) {
//                   if((n1 + n2 + j) < N  ||  (m1 + m2 + m3 + i) <  M) {
//                      C_p[ind_ob] = C[n2 + m3*N + i*N + j];
//                   }
//                   ind_ob++;
//                }
//             }
//          }
//       }

//    } else {

//       for(int n2 = 0; n2 < n_c; n2 += n_r) {
//          for(int m3 = 0; m3 < m_c; m3 += m_r) {
//             for(int i = 0; i < m_r; i++) {
//                for(int j = 0; j < n_r; j++) {
//                   C_p[ind_ob] = C[n2 + m3*N + i*N + j];
//                   ind_ob++;
//                }
//             }
//          }
//       }
//    }
// }




void pack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n) {

   int ind_ob = 0;

   if(pad_m || pad_n) {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  if((n1 + n2 + j) >= N  ||  (m1 + m2 + m3 + i) >=  M) {
                     C_p[ind_ob] = 0.0; // padding
                  } else {
                     C_p[ind_ob] = C[n2 + m3*N + i*N + j];
                  }
                  ind_ob++;
               }
            }
         }
      }

   } else {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  C_p[ind_ob] = C[n2 + m3*N + i*N + j];
                  ind_ob++;
               }
            }
         }
      }
   }
}




// // initialize an operation block of matrix A
// initialize an operation block of matrix A
void pack_ob_A_multiple_buf(float* A, float* A_p, int M, int K, int m1, int k1, int m2, int m_c, int k_c, int m_r, bool pad) {

   int ind2 = 0;
   
   if(pad) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < m_r; j++) {

               if((m1 + m2 + m3 + j) >=  M) {
                  A_p[ind2] = 0.0;
               } else {
                  A_p[ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
               }

               ind2++;
            }
         }
      }     
   } 

   else {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < k_c; i++) {
            for(int j = 0; j < m_r; j++) {
               A_p[ind2] = A[m1*K + k1 + m2*K + m3*K + i + j*K];
               ind2++;
            }
         }
      }
   }
}



void pack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r, bool pad) {

   int ind2 = 0;

   if(pad) {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  if((n1 + n2 + j) >= N  ||  (m1 + m2 + m3 + i) >=  M) {
                     C_p[ind2] = 0.0; // padding
                  } else {
                     C_p[ind2] = C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j];
                  }
                  ind2++;
               }
            }
         }
      }

   } else {

      for(int n2 = 0; n2 < n_c; n2 += n_r) {
         for(int m3 = 0; m3 < m_c; m3 += m_r) {
            for(int i = 0; i < m_r; i++) {
               for(int j = 0; j < n_r; j++) {
                  C_p[ind2] = C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j];
                  ind2++;
               }
            }
         }
      }
   }
}



void unpack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r) {

   int ind2 = 0;

   for(int n2 = 0; n2 < n_c; n2 += n_r) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < m_r; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) < N  &&  (m1 + m2 + m3 + i) <  M) {
                  C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j] = C_p[ind2];
               }
               ind2++;
            }
         }
      }
   }

}


// when writing reorderd rows back to C, 
// write to correct row row_inds[m1 + m2 + m3 + i]

void unpack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
            int m_c, int n_c, int m_r, int n_r) {

   int ind2 = 0;

   for(int n2 = 0; n2 < n_c; n2 += n_r) {
      for(int m3 = 0; m3 < m_c; m3 += m_r) {
         for(int i = 0; i < m_r; i++) {
            for(int j = 0; j < n_r; j++) {
               if((n1 + n2 + j) < N  &&  (m1 + m2 + m3 + i) <  M) {
                  // C[n1 + m1*N + m2*N + n2 + m3*N + i*N + j] = C_p[ind2];
                  C[n2 + m3*N + i*N + j] = C_p[ind2];
               }
               ind2++;
            }
         }
      }
   }
}

