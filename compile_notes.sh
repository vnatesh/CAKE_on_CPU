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


# compile BLIS test with MPI
mpigcc -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L \
-fopenmp -I/usr/local/include/blis -DBLIS_VERSION_STRING=\"0.8.0-13\" -I. -c blis_test.c -o blis_test.o
echo "Linking blis_test.x against '/usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt'"
mpigcc blis_test.o /usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt -o blis_test.x
rm blis_test.o


# compile ARMPL gemm test
gcc -I/opt/arm/armpl_20.3_gcc-7.1/include -fopenmp  arm_test.c -o test.o  \
/opt/arm/armpl_20.3_gcc-7.1/lib libarmpl_lp64_mp.a -L{ARMPL_DIR}/lib -lm -o arm_test
 

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

