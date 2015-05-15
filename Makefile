all: afnumpy/arrayfire.py

afnumpy/af_wrap.cxx: afnumpy/af.i
	swig -I/usr/local/include -c++ -python afnumpy/af.i

afnumpy/arrayfire.py: afnumpy/af_wrap.cxx afnumpy/multiarray.py
	python setup.py install
