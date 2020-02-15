from __future__ import print_function
from __future__ import division

# Load the original CPP Module
import NNcpp 

# Load the other modules
import minnie
import minnie.Atoms



class SymmetricFunctions(object):
    def __init__(self):
        """
        This class preserves all the modules of the standard symmetric function analisys.
        """

        # Create the Capsule object for the CPP SymmetricFunction class
        self._SymFunc = NNcpp.CreateSymFuncClass()

    def LoadFromCFG(self, fname):
        """
        Load from file
        ==============

        Load the symmetric function from a CFG file.

        Parameters
        ----------
            fname : string
                String to the position of the file.
        """

        # Call the C++ function to load the symmetric functions
        NNcpp.LoadSymmetricFunctionsFromCFG(self._SymFunc, fname)

    def SaveFromCFG(self, fname):
        """
        Save to file
        ==============

        Save the symmetric function into a CFG file.

        Parameters
        ----------
            fname : string
                String to the position of the file.
        """

        NNcpp.SaveSymFuncToCFG(self._SymFunc, fname)

    def GetSymmetricFunctions(self, atoms, Nx = 1, Ny = 1, Nz = 1):
        """
        GET SYMMETRIC FUNCTIONS
        =======================

        Computes the symmetric functions for the given atoms class.

        Parameters
        ----------
            atoms : minnie.Atoms.Atoms()
                The atoms to which you want to compute the symmetric functions.
            Nx, Ny, Nz : int
                The supercell to be created in which to compute the symmetric functions

        Results
        -------
            sym_funcs : ndarray( size = (N_atoms, N_syms), dtype = np.double)
                The symmetric functions for each atoms of the system.
        """

        sym_funcs = NNcpp.GetSymmetricFunctions(self._SymFunc, atoms._atoms, Nx, Ny, Nz)
        return sym_funcs