"""
The python setup to automatically compile the Neural Network source.
"""

from distutils.core import setup, Extension
import os

# Compiler flags
CPP = "/usr/bin/g++"
LIB_CONFIG = "config++"

# Setup custom environments
os.environ["CC"] = CPP
os.environ["CXX"] = CPP


NNcpp = Extension("NNcpp", ["src/atoms.cpp", "src/symmetric_functions.cpp", "src/ensemble.cpp", "src/p_interface.cpp"], 
                  libraries = [LIB_CONFIG])


setup(name = 'NNcpp', version = '0.01a', 
      description = """
      A deep neural network for electronic structure calculation.
      """, 
      author = 'Lorenzo Monacelli', 
      author_email = 'mesonepigreco@gmail.com',
      ext_modules = [NNcpp])