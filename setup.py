#!/usr/bin/env python

"""
setup.py file for afnumpy
"""

from distutils.core import setup



setup (name = 'afnumpy',
       version = '0.1',
       author      = "Filipe Maia",
       description = """A GPU-ready drop-in replacement for numpy""",
       packages = ["afnumpy", "afnumpy/core", "afnumpy/lib", "afnumpy/linalg"],
       )
