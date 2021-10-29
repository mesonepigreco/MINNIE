from __future__ import print_function
from __future__ import division 

import NNcpp
import numpy as np

"""
This file builds and create the Atoms class, wrapper of the C++ Atoms class
"""

__CC__ = False
try:
    import cellconstructor as CC, cellconstructor.Structure
    __CC__ = True
except:
    pass

class Atoms(object):

    def __init__(self, N_atoms, atom_class = None):
        """ 
        This class contains a system with atoms.

        Parameters
        ----------
            N_atoms : int 
                the total number of atoms
            atom_class : PyCapsule
                This is the PyCapsule.
                Avoid specifying this unless you really know what you are doing.
                Otherwise it will cause a Segmentation Fault.
        """
        self.N_atoms = N_atoms
        if atom_class is None:
            self._atoms = NNcpp.CreateAtomsClass(N_atoms)
        else:
            self._atoms = atom_class
            self.N_atoms = NNcpp.GetNAtoms(atom_class)
    
    

    def set_coords_types(self, coords, types, uc = None):
        """
        Set the atomic coordinates and types.
        coordinates must be a (nat, 3) array while types must be a list of atomic types
        """
        assert coords.shape[0] == self.N_atoms
        assert coords.shape[1] == 3

        assert len(types) == self.N_atoms

        new_coords = np.zeros(coords.shape, dtype = np.double, order = "C")
        new_coords[:,:] = coords
        new_types = np.zeros(types.shape, dtype = np.intc)
        new_types[:] = types
        new_uc = np.zeros( (3,3), dtype = np.double, order = "C")
        if uc:
            new_uc[:,:] = uc

        NNcpp.SetAtomsCoordsTypes(new_coords, types, uc)

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

        self.set_coords_types_uc(structure.coords, formatted_types, structure.unit_cell)

    def get_coords_types_uc(self):
        """
        Returns an array of the coordinates (Angstrom) and types (id)

        Returns
        -------
            coords : ndarray(size = (nat, 3))
                The cartesian position of the atoms
            types : ndarray (dtype = np.intc)
                The id that identifies the atomic species.
            unit_cell : ndarray(size = (3,3))
                The unit cell vectors
        """

        coords = np.zeros( (self.N_atoms, 3) , dtype = np.double, order = "C")
        uc = np.zeros( (3, 3) , dtype = np.double, order = "C")
        types = np.zeros(self.N_atoms, dtype = np.intc, order = "C")

        NNcpp.GetAtomsCoordsTypes(self._atoms, coords, types, uc)

        return coords, types, uc

    def get_cellconstructor(self, atom_label_dict):
        """
        GET THE CELLCONSTRUCTOR STRUCTURE
        =================================

        Get the structure of this atom into a cellconstructor Structure.

        Parameters
        ----------
            atom_label_dict : dict
                A dictionary that has for keys each ID of the atomic type and the corresponding value 
                is the periodic table symbol.

        Results
        -------
            struct : cellconstructor.Structure.Structure
                The structure of cellconstructor
        """

        if not __CC__:
            raise ImportError("Error, cellconstructor module not installed.")

        coords, types, uc = self.get_coords_types_uc()

        struct = CC.Structure.Structure(self.N_atoms)
        struct.coords[:,:] = coords
        struct.atoms = [atom_label_dict[x] for x in types]
        struct.has_unit_cell = True
        struct.unit_cell = uc

        return struct
        
