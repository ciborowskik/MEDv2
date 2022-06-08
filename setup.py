from setuptools import setup, Extension

module = Extension('cython_dist_merge_functions', sources=['cython_dist_merge_functions.pyx'])

setup(
    ext_modules=[module]
)
