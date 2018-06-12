#!/usr/bin/env python

"""
setup.py file for afnumpy
"""

from setuptools import setup
import setuptools.command.build_py
import subprocess
import os.path

version = '1.1'
name = 'afnumpy'
cwd = os.path.dirname(os.path.abspath(__file__))

def _get_version_hash():
    """Talk to git and find out the tag/hash of our latest commit"""
    try:
        p = subprocess.Popen(["git", "rev-parse", "--short" ,"HEAD"],
                             stdout=subprocess.PIPE)
    except EnvironmentError:
        print("Couldn't run git to get a version number for setup.py")
        return
    ver = p.communicate()[0]
    return ver.strip()

git_hash = _get_version_hash()

# Get the long description from the README file
with open(os.path.join(cwd, 'README.md')) as f:
    long_description = f.read()

long_description +=  "\n### Git commit\n[%s](http://github.com/FilipeMaia/afnumpy/commit/%s)" % (git_hash,git_hash)


class Version(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    def create_version_file(self):
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, name, 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))
            f.write("__githash__ = '{}'\n".format(git_hash))

setup (name = name,
       version = version,
       author      = "Filipe Maia",
       author_email = "filipe.c.maia@gmail.com",
       url = 'https://github.com/FilipeMaia/afnumpy',
       download_url = 'https://github.com/FilipeMaia/afnumpy/archive/'+version+'.tar.gz',
       keywords = ['arrayfire', 'numpy', 'GPU'],
       description = """A GPU-ready drop-in replacement for numpy""",
       long_description= long_description,
       long_description_content_type='text/markdown',
       packages = ["afnumpy", "afnumpy/core", "afnumpy/lib", "afnumpy/linalg"],
       install_requires=['arrayfire', 'numpy'],
       classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.2',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
       ],
       license='BSD',
       cmdclass={"version": Version},
       
)
