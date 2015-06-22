all: afnumpy/arrayfire.py

SWIGFLAGS = -c++ -python

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    SWIGFLAGS += -D__linux__
endif
ifeq ($(UNAME_S),Darwin)
    SWIGFLAGS += -D__APPLE__ -D__MACH__
endif

afnumpy/af_wrap.cxx: afnumpy/af.i
	swig -I/usr/local/include ${SWIGFLAGS} afnumpy/af.i

afnumpy/arrayfire.py: afnumpy/af_wrap.cxx afnumpy/multiarray.py
	python setup.py install

