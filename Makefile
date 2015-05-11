all: afnumpy/af_wrap.cxx

afnumpy/af_wrap.cxx: afnumpy/af.i
	swig -I/usr/local/include -c++ -python afnumpy/af.i
	python setup.py install
