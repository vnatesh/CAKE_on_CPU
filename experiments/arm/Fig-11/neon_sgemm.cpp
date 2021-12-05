/*
 * Copyright (c) 2018-2019 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"
#include <omp.h>
#include <cstdlib>

using namespace arm_compute;
using namespace utils;

class NESGEMMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        alpha = 1.0f;
        beta  = 0.0f;

        M = strtol(argv[1], nullptr, 10);
        N = strtol(argv[2], nullptr, 10);
        K = strtol(argv[3], nullptr, 10);
        p = atoi(argv[4]);

        printf("M = %ld, K = %ld, N = %ld, p = %d\n", M,K,N,p);
        
        src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
        src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
        src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));


        init_sgemm_output(dst, src0, src1, DataType::F32);

        // Configure function
        sgemm.configure(&src0, &src1, nullptr, &dst, alpha, beta);

        // Allocate all the images
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();

        src2.allocator()->allocate();

        fill_random_tensor(src0, -1.f, 1.f);
        fill_random_tensor(src1, -1.f, 1.f);
        fill_random_tensor(src2, -1.f, 1.f);
    
        // Dummy run for CLTuner
        struct timespec start, end;
        double diff_t;

        clock_gettime(CLOCK_REALTIME, &start);

        sgemm.run();

        clock_gettime(CLOCK_REALTIME, &end);
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
        printf("sgemm 1 time: %f \n", diff_t); 


        return true;
    }

    void do_run() override
    {
        // Execute the function
        struct timespec start, end;
        double diff_t;

        clock_gettime(CLOCK_REALTIME, &start);

        // use p cores for experiment
        NEScheduler::get().set_num_threads(p);
        sgemm.run();

        clock_gettime(CLOCK_REALTIME, &end);
        long seconds = end.tv_sec - start.tv_sec;
        long nanoseconds = end.tv_nsec - start.tv_nsec;
        diff_t = seconds + nanoseconds*1e-9;
        printf("sgemm time: %f \n", diff_t); 

        char fname[50];
        snprintf(fname, sizeof(fname), "results_sq");
        FILE *fp;
        fp = fopen(fname, "a");
        fprintf(fp, "armcl,%d,%ld,%f\n",p,M,diff_t);
        fclose(fp);

    }
    void do_teardown() override
    {
        if(!output_filename.empty()) /* Save to .npy file */
        {
            save_to_npy(dst, output_filename, is_fortran);
        }
    }

private:
    Tensor      src0{}, src1{}, src2{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    size_t      M;
    size_t      N;
    size_t      K;
    int         p;
    bool        is_fortran{};
    std::string output_filename{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NESGEMMExample>(argc, argv);
}