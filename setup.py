#!/usr/bin/env python

"""
setup.py file for afnumpy
"""

from distutils.core import setup
from afnumpy import __version__


setup (name = 'afnumpy',
       version = __version__,
       author      = "Filipe Maia",
       author_email = "filipe.c.maia@gmail.com",
       url = 'https://github.com/FilipeMaia/afnumpy',
       download_url = 'https://github.com/FilipeMaia/afnumpy/archive/'+__version__+'.tar.gz',
       keywords = ['arrayfire', 'numpy', 'GPU'],
       description = """A GPU-ready drop-in replacement for numpy""",
       packages = ["afnumpy", "afnumpy/core", "afnumpy/lib", "afnumpy/linalg"],
       install_requires=['arrayfire', 'numpy'],
)
