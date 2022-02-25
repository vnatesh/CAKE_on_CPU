#
#
#  compilation notes for static and dynamic linking in BLIS, CAKE, MKL, and ARM experiments   
#  
#  


# intel MKL link advisor to generate compilation command given system parameters
# https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html

# intel vtune standalone install
# https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#vtune


# dynamic linking of MKL integer GEMM benchmark on macOS
 gcc -DMKL_ILP64 -m64  -I"${MKLROOT}/include" matmul_int.c -L${MKLROOT}/lib \
 -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core \
 -liomp5 -lpthread -lm -ldl -o matmul_int


# dynamic linking of MKL sgemm test
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c -Wl,--no-as-needed \
-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread \
-lm -ldl -o mkl_sgemm_test


# compile BLIS test program
gcc -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L \
-fopenmp -I/usr/local/include/blis -DBLIS_VERSION_STRING=\"0.8.0-13\" -I. -c blis_test.c -o blis_test.o
echo "Linking blis_test.x against '/usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt'"
gcc blis_test.o /usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt -o blis_test.x
rm blis_test.o

export BLIS_NUM_THREADS=10



# compile blis on alienware (haswell)
gcc -g -O2 -std=c99 -Wall -Wno-unused-function -Wfatal-errors -fPIC  -D_POSIX_C_SOURCE=200112L -fopenmp -I/tmp/CAKE_on_CPU/include/blis -DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o
echo "Linking blis_test against '/tmp/CAKE_on_CPU/blis/lib/haswell/libblis.a  -lm -lpthread -fopenmp -lrt'"
g++ blis_test.o /tmp/CAKE_on_CPU/blis/lib/haswell/libblis.a  -lm -lpthread -fopenmp -lrt -o blis_test


# compile blis on raspberry pi 3b+ (coretx a53)
gcc -g -O2 -std=c99 -Wall -Wno-unused-function -Wfatal-errors -fPIC  -D_POSIX_C_SOURCE=200112L -fopenmp -I/home/ubuntu/CAKE_on_CPU/include/blis -DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o
echo "Linking blis_test against '/home/ubuntu/CAKE_on_CPU/blis/lib/cortexa53/libblis.a  -lm -lpthread -fopenmp -lrt'"
g++ blis_test.o /home/ubuntu/CAKE_on_CPU/blis/lib/cortexa53/libblis.a  -lm -lpthread -fopenmp -lrt -o blis_test


# compile blis on raspberry pi 4 (cortex a72)
gcc -g -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC  -D_POSIX_C_SOURCE=200112L -fopenmp -I/home/ubuntu/CAKE_on_CPU/include/blis -DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o
echo "Linking blis_test against '/home/ubuntu/CAKE_on_CPU/blis/lib/cortexa57/libblis.a  -lm -lpthread -fopenmp -lrt'"
g++ blis_test.o /home/ubuntu/CAKE_on_CPU/blis/lib/cortexa57/libblis.a  -lm -lpthread -fopenmp -lrt -o blis_test

# compile armpl test raspberry pi 4
gcc -I/opt/arm/armpl_21.1_gcc-9.3/include -fopenmp  arm_test.c -o test.o  /opt/arm/armpl_21.1_gcc-9.3/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib -lm -o arm_test;

# compile armpl test raspberry pi 3b+
gcc -I/opt/arm/armpl_21.0_gcc-10.2/include -fopenmp  arm_test.c -o test.o  /opt/arm/armpl_21.0_gcc-10.2/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib -lm -o arm_test;


# compile BLIS test with MPI
mpigcc -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L \
-fopenmp -I/usr/local/include/blis -DBLIS_VERSION_STRING=\"0.8.0-13\" -I. -c blis_test.c -o blis_test.o
echo "Linking blis_test.x against '/usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt'"
mpigcc blis_test.o /usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt -o blis_test.x
rm blis_test.o


# compile ARMPL gemm test
gcc -I/opt/arm/armpl_20.3_gcc-7.1/include -fopenmp  arm_test.c -o test.o  \
/opt/arm/armpl_20.3_gcc-7.1/lib libarmpl_lp64_mp.a -L{ARMPL_DIR}/lib -lm -o arm_test
 

# build ARM Compute Library
git clone https://github.com/ARM-software/ComputeLibrary.git;
cd ComputeLibrary;
scons -j4 Werror=0 pmu=1 openmp=1 neon=1 opencl=0 os=linux arch=arm64-v8a;


# compille ARM sgemm example on raspberry pi 4 (NEON)
aarch64-linux-gnu-g++ -o neon_sgemm.o -c -Wall -DARCH_ARM -Wextra -pedantic \
-Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default \
-std=c++14 -Woverloaded-virtual -Wformat-security -Wctor-dtor-privacy -Wsign-promo \
-Weffc++ -Wno-overlength-strings -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -C \
-fopenmp -march=armv8-a -DENABLE_NEON -DARM_COMPUTE_ENABLE_NEON -Wno-ignored-attributes \
-DENABLE_FP16_KERNELS -DENABLE_FP32_KERNELS -DENABLE_QASYMM8_KERNELS \
-DENABLE_QASYMM8_SIGNED_KERNELS -DENABLE_QSYMM16_KERNELS -DENABLE_INTEGER_KERNELS \
-DENABLE_NHWC_KERNELS -DENABLE_NCHW_KERNELS -O3 -D_GLIBCXX_USE_NANOSLEEP \
-DARM_COMPUTE_CPP_SCHEDULER=1 -DARM_COMPUTE_OPENMP_SCHEDULER=1 \
-DARM_COMPUTE_GRAPH_ENABLED -DARM_COMPUTE_CPU_ENABLED \
-I/home/ubuntu/ComputeLibrary/include -I/home/ubuntu/ComputeLibrary \
-I/home/ubuntu/ComputeLibrary neon_sgemm.cpp

aarch64-linux-gnu-g++ -o neon_sgemm -fopenmp neon_sgemm.o \
/home/ubuntu/ComputeLibrary/build/utils/Utils.o -L/home/ubuntu/ComputeLibrary/build \
-L/home/ubuntu/ComputeLibrary -lpthread -larm_compute -larm_compute_core


#static linking MKL
gcc  -DMKL_ILP64 -m64 -I${MKLROOT}/include test.c -Wl,--no-export-dynamic \
-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a \
/usr/lib/gcc/x86_64-linux-gnu/5/libgomp.a  -Wl,--end-group -Wl,--whole-archive \
-lpthread -Wl,--no-whole-archive -static  /usr/lib/x86_64-linux-gnu/libc.a -static \
/usr/lib/x86_64-linux-gnu/libdl.a -static /usr/lib/x86_64-linux-gnu/libm.a

gcc -DMKL_ILP64 -m64 -I${MKLROOT}/include mkl_sgemm_test.c -Wl,--no-export-dynamic -Wl,--start-group \
${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
${MKLROOT}/lib/intel64/libmkl_core.a /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.a  \
-Wl,--end-group -static /usr/lib/x86_64-linux-gnu/libpthread.a -static  \
/usr/lib/x86_64-linux-gnu/libc.a -static /usr/lib/x86_64-linux-gnu/libdl.a \
-static /usr/lib/x86_64-linux-gnu/libm.a -o mkl_sgemm_test


# dynamic compilation of CAKE with BLIS kernel
g++ -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -D_POSIX_C_SOURCE=200112L \
-fopenmp -I/usr/local/include/blis -DBLIS_VERSION_STRING=\"0.8.1-8\" -I. \
/tmp/CAKE_on_CPU/src/cake_sgemm_test.cpp /tmp/CAKE_on_CPU/src/block_sizing.cpp \
/tmp/CAKE_on_CPU/src/cake_sgemm.cpp /tmp/CAKE_on_CPU/src/pack.cpp src/util.cpp \
/tmp/CAKE_on_CPU/src/unpack.cpp /usr/local/lib/libblis.a -lm -lpthread -fopenmp \
-lrt -o cake_sgemm_test.x


# compile sparse GEMM tests in MKL 
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
rm -rf prof_result
make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu 


# run vtune profiling on sparse GEMM
vtune --collect memory-access -data-limit=0 \
-result-dir=/opt/intel/oneapi/mkl/2021.1.1/examples/sycl/prof_result \
/opt/intel/oneapi/mkl/2021.1.1/examples/sycl/_results/intel64_so_tbb/spblas/sparse_gemm.out 


# run perf and socwatch power reports
perf stat -a -e "power/energy-pkg/","power/energy-ram/" -o \
power_reports/cake_M$k-K500-N500-$j ./cake_sgemm_test $k 500 500;

/opt/intel/oneapi/vtune/2021.1.1/socwatch/x64/socwatch -t 20 -f ddr-bw -o \
power_reports/cake_socw_M$k-K$K-N$N-$j -p ./cake_sgemm_test $k $K $N;





#creates  shared library libcake.so
g++ -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -D_POSIX_C_SOURCE=200112L \
-I/usr/local/include/blis -DBLIS_VERSION_STRING=\"0.8.1-8\" -I/tmp/CAKE_on_CPU/include \
/tmp/CAKE_on_CPU/src/block_sizing.cpp /tmp/CAKE_on_CPU/src/cake_sgemm.cpp \
/tmp/CAKE_on_CPU/src/pack.cpp src/util.cpp /tmp/CAKE_on_CPU/src/unpack.cpp \
/usr/local/lib/libblis.a -lm -lpthread -fopenmp -lrt -shared -o libcake.so


# now that library is created above, compile test sgemm file from 
# anywhere on machine as long as you specify location of 
# cake.h file (-I/tmp/CAKE_on_CPU/src)

g++ -I/usr/local/include/blis -I/tmp/CAKE_on_CPU/include -L/tmp/CAKE_on_CPU \
-Wall -o testing cake_sgemm_test.cpp -lcake

# add this to env.sh
LD_LIBRARY_PATH=$CAKE_HOME:$LD_LIBRARY_PATH

