"""
The python setup to automatically compile the Neural Network source.
"""

from distutils.core import setup, Extension
import os
import numpy as np

# Compiler flags
CPP = "g++"
LIB_CONFIG = "config++"
ALL_LIBS = [LIB_CONFIG, "gsl", "gslcblas", "boost_system", "boost_filesystem"]
EXTRA_COMPILE_FLAGS = ["--std=c++11"]
# Setup custom environments
os.environ["CC"] = CPP
#os.environ["CXX"] = CPP

try:
      import mpi4py 
      if os.environ["MPICXX"]:
            print("Parallel environment detected succesfully.")
            print("Activating the parallelization")
            os.environ["CC"] = os.environ["MPICXX"]
            os.environ["CXX"] = os.environ["MPICXX"]
            os.environ["LDSHARED"] = os.environ["MPICXX"] + " -shared"
            EXTRA_COMPILE_FLAGS += ["-D_MPI"]
      else:
            print()
            print("Error, mpi4py found but the einvironment variable MPICXX is not set.")
            print("Parallel calculation is disabled.")
            print("To activate parallel calculation, run the code specifying the c++ MPI compiler as:")
            print("   MPICXX=mpic++ python setup.py install")
            print()


except:
      print()
      print("Warning, mpi4py module not found!")
      print("The parallelization is disable")
      print("Please, install mpi4py (and MPI C compiler)")
      print("Then re-install this package to enable the parallelization.")
      print()


all_nnfiles = ["src/AtomicNetwork.cpp", 
               "src/analyze_ensemble.cpp",
               "src/atoms.cpp",
               "src/ensemble.cpp",
               "src/matrix.c",
               "src/network.cpp",
               "src/symmetric_functions.cpp",
               "src/Trainer.cpp",
               "src/utils.cpp",
               "src/p_interface.cpp"] # The p_interface.cpp is the main for the python interface

print("List of sources:")
print(all_nnfiles)
NNcpp = Extension("NNcpp", all_nnfiles, 
                  libraries = ALL_LIBS,
                  include_dirs = [np.get_include()],
                  extra_compile_args = EXTRA_COMPILE_FLAGS)


setup(name = 'minnie', version = '0.01a', 
      description = """
      A deep neural network for electronic structure calculation.
      """, 
      author = 'Lorenzo Monacelli', 
      author_email = 'mesonepigreco@gmail.com',
      packages = ["minnie"],
      package_dir = {"minnie": "pythonmodule"},
      ext_modules = [NNcpp])
