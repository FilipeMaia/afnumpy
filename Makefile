all: afnumpy/arrayfire.py

SWIGFLAGS = -c++ -python

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    SWIGFLAGS += -D__linux__
endif
ifeq ($(UNAME_S),Darwin)
    SWIGFLAGS += -D__APPLE__ -D__MACH__
endif
ifdef CPLUS_INCLUDE_PATH
    SWIGFLAGS += -I${CPLUS_INCLUDE_PATH}
endif

afnumpy/af_wrap.cxx: afnumpy/af.i
	swig -I/usr/local/include -I${CPLUS_INCLUDE_PATH} ${SWIGFLAGS} afnumpy/af.i

afnumpy/arrayfire.py: afnumpy/af_wrap.cxx afnumpy/multiarray.py
	python setup.py install

