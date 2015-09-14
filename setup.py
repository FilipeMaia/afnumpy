#!/usr/bin/env python

"""
setup.py file for afnumpy
"""

from distutils.core import setup



setup (name = 'afnumpy',
       version = '1.0b1',
       author      = "Filipe Maia",
       author_email = "filipe.c.maia@gmail.com",
       url = 'https://github.com/FilipeMaia/afnumpy',
       download_url = 'https://github.com/afnumpy/tarball/v1.0b1',
       keywords = ['arrayfire', 'numpy', 'GPU'],
       description = """A GPU-ready drop-in replacement for numpy""",
       packages = ["afnumpy", "afnumpy/core", "afnumpy/lib", "afnumpy/linalg"],
       install_requires=['arrayfire'],
)
