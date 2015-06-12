#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import os
import numpy

os.environ["ARCHFLAGS"]="-arch x86_64" 
arrayfire_module = Extension('_arrayfire',
                             sources=['afnumpy/af_wrap.cxx'],
                             libraries = ['afcuda'],
                             runtime_library_dirs = ['/usr/local/lib'],
                             extra_link_args = ['-Wl,-rpath,/usr/local/lib'],
                             extra_compile_args = ['-std=c++11','-g','-O0','-fno-omit-frame-pointer'],
                         )

setup (name = 'afnumpy',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       include_dirs = [numpy.get_include()],
       ext_modules = [arrayfire_module],
       packages = ["afnumpy"],
       )
