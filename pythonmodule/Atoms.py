from __future__ import print_function
from __future__ import division 

import NNcpp

"""
This file builds and create the Atoms class, wrapper of the C++ Atoms class
"""


class Atoms(object):

    def __init__(self, N_atoms):
        """ 
        This class contains a system with atoms.
        """
        
        self._atoms = NNcpp.CreateAtomsClass(N_atoms)