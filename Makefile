#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# Makefile
#
# Field G. Van Zee
#
# Makefile for BLIS typed API example code.
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all install build clean



#
# --- Determine makefile fragment location -------------------------------------
#

# Comments:
# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
ifneq ($(strip $(BLIS_INSTALL_PATH)),)
LIB_PATH   := $(BLIS_INSTALL_PATH)/lib
INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
SHARE_PATH := $(BLIS_INSTALL_PATH)/share/blis
else
DIST_PATH  := blis
LIB_PATH    = blis/lib/$(CONFIG_NAME)
INC_PATH    = blis/include/$(CONFIG_NAME)
SHARE_PATH := ../..
endif



#
# --- Include common makefile definitions --------------------------------------
#

# Include the common makefile fragment.
-include $(SHARE_PATH)/common.mk



#
# --- General build definitions ------------------------------------------------
#

INCLUDE_PATH  := $(CAKE_HOME)/include
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(sort $(patsubst $(INCLUDE_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(INCLUDE_PATH)/*.c)))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
# CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))
CFLAGS_tmp := -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L -lpthread -fopenmp
CFLAGS_tmp        += -I$(INCLUDE_PATH) 
CFLAGS_tmp        += -g
# Add local header paths to CFLAGS

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# shared library name.
LIBCAKE      := libcake.so

CAKE_SRC := $(CAKE_HOME)/src
UNAME_P := $(shell uname -p)
SRC_FILES =  $(wildcard $(CAKE_HOME)/src/*.cpp)
SRC_FILES := $(filter-out $(CAKE_HOME)/src/linear.cpp, $(SRC_FILES)) 

ifeq ($(UNAME_P),aarch64)
	KERNELS = $(CAKE_SRC)/kernels/armv8/*.cpp
	TARGETS = cake_armv8
	CFLAGS_tmp += -O3
else ifeq ($(UNAME_P),x86_64)
	KERNELS = $(CAKE_SRC)/kernels/haswell/*.cpp
	CFLAGS_tmp += -mavx -mfma
	TARGETS = cake_haswell
	CFLAGS_tmp += -O2
else
	TARGETS = cake_blis
	CFLAGS_tmp  := $(call get-user-cflags-for,$(CONFIG_NAME))
	CFLAGS_tmp += -I$(CAKE_HOME)/include/blis -I$(CAKE_HOME)/include
endif



CFLAGS 	:= $(filter-out -std=c99, $(CFLAGS_tmp))


LIBS ?=
#LIBDIR += -L. -L$(SYSTEMC_HOME)/lib-linux64 -L$(BOOST_HOME)/lib
LIBS += $(BLIS_INSTALL_PATH)/lib/libblis.a 

#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: $(TARGETS) 

install:
	./install.sh
	
# --- Environment check rules ---

check-env: check-env-make-defs check-env-fragments check-env-config-mk

check-env-config-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
    $(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-env-make-defs: check-env-fragments
ifeq ($(MAKE_DEFS_MK_PRESENT),no)
    $(error Cannot proceed: make_defs.mk not detected! Invalid configuration)
endif

# 	dpcpp -fp-speculation=fast g++ $(CFLAGS) $(CAKE_SRC)/block_sizing.cpp $(CAKE_SRC)/cake_sgemm.cpp \

cake_blis: $(wildcard *.h) $(wildcard *.c) 
	g++ $(CFLAGS) $(CFLAGS) $(SRC_FILES) $(LIBS) \
	$(LDFLAGS) -DUSE_BLIS -shared -o $(LIBCAKE)

cake_haswell: $(wildcard *.h) $(wildcard *.c)
	g++  $(CFLAGS) $(SRC_FILES) $(KERNELS) \
	$(LDFLAGS) -DUSE_CAKE_HASWELL -shared -o $(LIBCAKE)

cake_armv8: $(wildcard *.h) $(wildcard *.c)
	g++ $(CFLAGS) $(SRC_FILES) $(KERNELS) \
	$(LDFLAGS) -DUSE_CAKE_ARMV8  -shared -o $(LIBCAKE)

# -- Clean rules --

clean:
	rm -rf *.o *.so
