# coding=utf-8
'''
Created on 2018/02/24

@author: snakagawa
'''
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

from numpy import get_include

setup(
      ext_modules = [Extension("", ["src/sgt/_cpolarize.pyx", "src/sgt/_crefine.pyx"], include_dirs=[get_include()]), 
                     ], 
      include_dirs=[get_include()], 
      cmdclass={'build_ext': build_ext}
)