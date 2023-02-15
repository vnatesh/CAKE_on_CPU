.PHONY: all install build clean


INCLUDE_PATH  := $(CAKE_HOME)/include
TEST_OBJ_PATH  := .


# Use the "framework" CFLAGS for the configuration family.
# CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))
CFLAGS_tmp := -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L 
CFLAGS_tmp        += -I$(INCLUDE_PATH) -g


# shared library name.
LIBCAKE      := libcake_kernels.so

CAKE_SRC := $(CAKE_HOME)/src
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),aarch64)
	KERNELS = $(CAKE_SRC)/kernels/armv8/dense.cpp
# 	KERNELS := $(filter-out $(CAKE_SRC)/kernels/armv8/blis_pack_armv8.cpp, $(KERNELS)) 
	TARGETS = cake_armv8
	CFLAGS_tmp += -O3 -mtune=cortex-a53
else ifeq ($(UNAME_M),x86_64)
	KERNELS = $(CAKE_SRC)/kernels/haswell/dense.cpp
# 	KERNELS := $(filter-out $(CAKE_SRC)/kernels/haswell/blis_pack_haswell.cpp, $(KERNELS)) 
	CFLAGS_tmp += -mavx -mfma -mtune=haswell
	TARGETS = cake_haswell
	CFLAGS_tmp += -O2
endif


CFLAGS 	:= $(filter-out -std=c99, $(CFLAGS_tmp))


# --- Primary targets ---

all: $(TARGETS) 

install:
	./install.sh
	
# 	dpcpp -fp-speculation=fast g++ $(CFLAGS) $(CAKE_SRC)/block_sizing.cpp $(CAKE_SRC)/cake_sgemm.cpp \

cake_haswell: $(wildcard *.h) $(wildcard *.c)
	g++  $(CFLAGS) $(KERNELS) \
	$(LDFLAGS) -DUSE_CAKE_HASWELL -shared -o $(LIBCAKE)

cake_armv8: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS) $(KERNELS) \
	$(LDFLAGS) -DUSE_CAKE_ARMV8 -shared -o $(LIBCAKE)

# -- Clean rules --

clean:
	rm -rf *.o *.so
