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
        self.N_atoms = N_atoms
        self._atoms = NNcpp.CreateAtomsClass(N_atoms)
    
    def set_coords_types(self, coords, types):
        """
        Set the atomic coordinates and types.
        coordinates must be a (nat, 3) array while types must be a list of atomic types
        """
        assert coords.shape[0] == N_atoms
        assert coords.shape[1] == 3

        assert len(types) == N_atoms

        new_coords = np.zeros(coords.shape, dtype = np.double, order = "C")
        NNcpp.set_coords_types(new_coords, types)

    def set_from_cellconstructor(self, structure):
        """
        Setup the atoms object from the cellconstructor structure
        """

        self._atoms = NNcpp.CreateAtomsClass(structure.N_atoms)

        types = np.unique(structure.atoms)
        dicttypes = {}
        for i, t in enumerate(types):
            dicttypes[t] = i 

        formatted_types = np.zeros(self.N_atoms, dtype = np.intc)
        formatted_types[:] = [dicttypes[x] for x in self.atoms]

        self.set_coords_types(structure.coords, formatted_types)