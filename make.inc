# Edit the following variables to link the correct library

# Please, specify here the C++ compiler (if different from g++)
CPP=g++ --std=c++11 -O3

# LIBCONFIG configuration settings
# Please, specify the cflags for the libconfig++ library (the include path)
INCCONFIG=$(pkg-config --cflags libconfig++)
LDCONFIG=`pkg-config --libs libconfig++` -lm

# Specify the GSL (GNU Scientific Library) linking flags
GSLFLAGS = -lgsl -lgslcblas
# Specify the boost linking flags
BOOSTFLAGS = -lboost_system -lboost_filesystem


CFLAGS=$(INCCONFIG)
LIBS=$(LDCONFIG)
