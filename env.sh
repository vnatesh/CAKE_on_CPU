#!/bin/bash

# configure environment variables
export BLIS_INSTALL_PATH=$PWD
export CAKE_HOME=$PWD
LD_LIBRARY_PATH=$CAKE_HOME:$LD_LIBRARY_PATH
