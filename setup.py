#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import numpy

arrayfire_module = Extension('_arrayfire',
                             sources=['afnumpy/af_wrap.cxx'],
                             libraries = ['afcuda'],
                             runtime_library_dirs = ['/usr/local/lib'],
                             extra_link_args = ['-Wl,-rpath,/usr/local/lib'],
                         )

setup (name = 'afnumpy',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       include_dirs = [numpy.get_include()],
       ext_modules = [arrayfire_module],
       packages = ["afnumpy"],
       )
