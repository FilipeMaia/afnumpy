#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


arrayfire_module = Extension('_arrayfire',
                             sources=['af_wrap.cxx'],
                             libraries = ['afcuda'],
                         )

setup (name = 'arrayfire',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [arrayfire_module],
       py_modules = ["arrayfire"],
       )
