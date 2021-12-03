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
CFLAGS_tmp         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS_tmp        += -I$(INCLUDE_PATH)
CFLAGS 	:= $(filter-out -fopenmp -std=c99, $(CFLAGS_tmp))

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# shared library name.
LIBCAKE      := libcake.so

CAKE_SRC := $(CAKE_HOME)/src

LIBS ?=
#LIBDIR += -L. -L$(SYSTEMC_HOME)/lib-linux64 -L$(BOOST_HOME)/lib
LIBS += $(BLIS_INSTALL_PATH)/lib/libblis.a 

#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: build

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


build: $(wildcard *.h) $(wildcard *.c) 
	g++ $(CFLAGS) $(CAKE_SRC)/block_sizing.cpp $(CAKE_SRC)/cake_sgemm.cpp \
	$(CAKE_SRC)/cake_sgemm_k_first.cpp $(CAKE_SRC)/cake_sgemm_m_first.cpp $(CAKE_SRC)/cake_sgemm_n_first.cpp \
	$(CAKE_SRC)/pack_helper.cpp $(CAKE_SRC)/pack_ob.cpp $(CAKE_SRC)/util.cpp \
	$(CAKE_SRC)/pack_k_first.cpp $(CAKE_SRC)/pack_m_first.cpp $(CAKE_SRC)/pack_n_first.cpp \
	$(CAKE_SRC)/unpack_k_first.cpp $(CAKE_SRC)/unpack_m_first.cpp $(CAKE_SRC)/unpack_n_first.cpp \
	$(LIBS) $(LDFLAGS) -shared -o $(LIBCAKE)


# -- Clean rules --

clean:
	rm -rf *.o *.so


# .PHONY: all install build clean



# INCDIR ?=
# #INCDIR += -I. -I$(SYSTEMC_HOME)/include -I$(BOOST_HOME)/include -I$(CATAPULT_HOME)/Mgc_home/shared/include
# INCDIR += -I$(BLIS_INSTALL_PATH)/include/blis -DBLIS_VERSION_STRING=\"0.8.0-13\" -I.


# # CFLAGS ?= 
# # CFLAGS += -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 \
# # 			-D_POSIX_C_SOURCE=200112L $(INCDIR)
# CFLAGS := $(call get-user-cflags-for,$(CONFIG_NAME))
# CFLAGS += $(INCDIR) 

# LIBS ?=
# #LIBDIR += -L. -L$(SYSTEMC_HOME)/lib-linux64 -L$(BOOST_HOME)/lib
# LIBS += $(BLIS_INSTALL_PATH)/lib/libblis.a -lm -lpthread -fopenmp -lrt 




# # gcc -O3 -O2 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99 
# # -D_POSIX_C_SOURCE=200112L -fopenmp -I/usr/local/include/blis -DBLIS_VERSION_STRING=\"0.8.0-13\" 
# # -I. *.c -o cake_sgemm_test.o /usr/local/lib/libblis.a  -lm -lpthread -fopenmp -lrt -o cake_sgemm_test.x


# all: build

# install:
# 	./install.sh

# build: $(wildcard *.h) $(wildcard *.c) 
# 	gcc $(CFLAGS) *.c -o cake_sgemm_test.o $(LIBS) -o cake_sgemm_test.x

# clean:
# 	rm -rf *.o
