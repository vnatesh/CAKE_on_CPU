#include <torch/torch.h>
#include <vector>
#include <torch/extension.h>
#include <iostream>

#include "cake.h"

#include <sys/time.h> 
#include <time.h> 


torch::Tensor linear_forward(torch::Tensor& cache_sz, torch::Tensor& input,  torch::Tensor& weight, const torch::Tensor& bias={}) {
  // auto output_check = at::matmul(input, weight.t());
  // if (bias.defined()) {
  //   output_check.add_(bias);
  // }
  int M = input.size(0);
  int N = weight.size(0);
  int K = input.size(1);
  // printf("pytorch M = %d, N = %d, K = %d\n", M,N,K);
  input = input.reshape({M*K}).contiguous();
  weight = weight.t().reshape({N*K}).contiguous();
  float* input_c = (float*) input.data_ptr<float>();
  float* weight_c = (float*) weight.data_ptr<float>();
  float* output_c = (float*) calloc(M * N , sizeof( float ));
  int p = 10; // number of cores to use

 // for(int i = 0; i < K; i++) {
 //    for(int j = 0; j < N; j++) {
 //      printf("%f ", weight_c[i*N + j]);
 //    }
 //    printf("\n");
 //  }
 //  printf("\n\n");

  // cake_sgemm call
  cake_cntx_t* cake_cntx = cake_query_cntx_torch(cache_sz[0].item<int>(), cache_sz[1].item<int>());
  cake_sgemm(input_c, weight_c, output_c, M, N, K, p, cake_cntx);

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto output = torch::from_blob((void*) output_c, {M*N}, options);
  output = output.reshape({M,N});

  if (bias.defined()) {
    output.add_(bias);
  }

  return output;
}


std::vector<torch::Tensor> linear_backward(torch::Tensor& cache_sz, torch::Tensor& grad_output, torch::Tensor& input, torch::Tensor& weight, const torch::Tensor& bias) {


  cake_cntx_t* cake_cntx = cake_query_cntx_torch(cache_sz[0].item<int>(), cache_sz[1].item<int>());

  int M = grad_output.size(0);
  int N = weight.size(1);
  int K = grad_output.size(1);
  grad_output = grad_output.reshape({M*K}).contiguous();
  weight = weight.reshape({N*K}).contiguous();
  float* grad_output_c = (float*) grad_output.data_ptr<float>();
  float* weight_c = (float*) weight.data_ptr<float>();
  float* grad_input_c = (float*) calloc(M * N , sizeof( float ));
  int p = 10; // number of cores to use

  cake_sgemm(grad_output_c, weight_c, grad_input_c, M, N, K, p, cake_cntx);


  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto grad_input = torch::from_blob((void*) grad_input_c, {M*N}, options);
  grad_input = grad_input.reshape({M,N});

  grad_output = grad_output.reshape({M,K}).t();
  int M1 = grad_output.size(0);
  int K1 = input.size(0);
  int N1 = input.size(1);
  grad_output = grad_output.reshape({M1*K1}).contiguous();
  grad_output_c = (float*) grad_output.data_ptr<float>();
  input = input.reshape({K1*N1}).contiguous();
  float* input_c = (float*) input.data_ptr<float>();
  float* grad_weight_c = (float*) calloc(M1 * N1 , sizeof( float ));

  cake_sgemm(grad_output_c, input_c, grad_weight_c, M1, N1, K1, p, cake_cntx);

  options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto grad_weight = torch::from_blob((void*) grad_weight_c, {M1*N1}, options);
  grad_weight = grad_weight.reshape({M1,N1});

  grad_output = grad_output.reshape({K1,M1});
  auto grad_bias = bias.defined() ? grad_output.sum(0, /*keepdim=*/false) : torch::Tensor{};

  // auto grad_input = at::matmul(grad_output, weight);
  // auto grad_weight = at::matmul(grad_output.t(), input);
  // auto grad_bias = bias.defined() ? grad_output.sum(0, /*keepdim=*/false) : torch::Tensor{};

  return {
    grad_input,
    grad_weight,
    grad_bias
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "linear forward");
  m.def("backward", &linear_backward, "linear backward");
}



