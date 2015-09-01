all: afnumpy/arrayfire.py

afnumpy/arrayfire.py: afnumpy/multiarray.py
	python setup.py install --user

