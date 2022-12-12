# coding="utf-8"

from setuptools import setup, Extension
from numpy import get_include

setup_args = dict(
    ext_modules = [
        Extension(
            'sgt._cpolarize',
            ['src/sgt/_cpolarize.c'],
            include_dirs = [get_include()]
        ), 
        Extension(
            'sgt._crefine',
            ['src/sgt/_crefine.c'],
            include_dirs = [get_include()]
        )
    ]
)
setup(**setup_args)