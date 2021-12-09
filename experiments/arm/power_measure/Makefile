.PHONY: all compile clean

ifneq ($(strip $(BLIS_INSTALL_PATH)),)
BLIS_INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
endif

# g++ cake_sgemm_test.cpp -I/usr/local/include/blis -I/tmp/CAKE_on_CPU/include -L/tmp/CAKE_on_CPU  -lcake -o testing

CAKE_INC_PATH  := $(CAKE_HOME)/include
TEST_OBJ_PATH  := .

# Use the "framework" CFLAGS for the configuration family.
CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS_tmp        += -I$(CAKE_INC_PATH)
CFLAGS_tmp		  += -I$(BLIS_INC_PATH)
CFLAGS 	:= $(filter-out -fopenmp -std=c99, $(CFLAGS_tmp))

# cake shared library
LIBCAKE      := -L$(CAKE_HOME) -lcake
CAKE_SRC := $(CAKE_HOME)/src


# --- Targets/rules ------------------------------------------------------------

all: cake armpl blis

cake: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS) cake_sgemm_test.cpp $(LIBCAKE) -o cake_sgemm_test

armpl:
	gcc -I/opt/arm/armpl_21.1_gcc-9.3/include -fopenmp  arm_test.c \
	-o test.o  /opt/arm/armpl_21.1_gcc-9.3/lib/libarmpl_lp64_mp.a \
	 -L{ARMPL_DIR}/lib -lm -o arm_test;

blis:
	gcc -g -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC  \
	-D_POSIX_C_SOURCE=200112L -fopenmp -I$(CAKE_HOME)/include/blis \
	-DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o; 
	g++ blis_test.o $(CAKE_HOME)/blis/lib/cortexa57/libblis.a \
	 -lm -lpthread -fopenmp -lrt -o blis_test;

clean:
	rm -rf *.o *.so

