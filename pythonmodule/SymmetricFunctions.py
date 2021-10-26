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

    def get_parameters(self, index, tipology = "g2"):
        """
        Get the symmetric function parameters.
        It returns a dictionary of the parameters

        Parameters
        ----------
            index : int
                The position of the symmetric function
            tipology : string
                either g2 or g4 for the respective type of symmetric functions.

        Returns
        -------
            parameters : dict
                The dictionary with the parameters and the respective values
        """
        assert tipology.lower().strip() in ["g2", "g4"], "Error accepted only 'g2' or 'g4' for the tipology."

        isg2 = 1
        if tipology.lower().strip() == "g4":
            isg2 = 0

        resuts = NNcpp.GetSymmetricFunctionParameters(self._SymFunc, index, isg2)    
        parameters = {}
        if isg2:
            parameters["g2_Rs"] = results[0]
            parameters["g2_eta"] = results[1]
        else:
            parameters["g4_zeta"] = results[0]
            parameters["g4_eta"] = results[1]
            parameters["g4_lambda"] = results[2]

        return parameters 

    def set_parameters(self, index, params):
        """
        Set the parameters of the symmetric function at the given index.
        The g2 or g4 function is inferred from keys of params

        Parameters
        ----------
            index : int
                The index of the symmetric function to be set.
            params : dict
                The parameters. allowed values are g2_Rs, g2_eta for g2 functions and 
                g4_zeta, g4_eta and g4_lambda for g4 functions.
        """

        isg2 = -1

        for k in params:
            if "g2" in k:
                if isg2 == 0:
                    raise ValueError("Error, only one kind between g2 and g4 can be specified.")
                isg2 = 1
            elif "g4" in k:
                if isg2 == 1:
                    raise ValueError("Error, only one kind between g2 and g4 can be specified.")
                isg2 = 0
            else:
                raise ValueError("Error, unknown parameter {}".format(k))

            
        #TODO
        pass
    
    def get_number_of_g2(self):
        n2, n4 = NNcpp.GetNSyms(self._SymFunc)
        return n2

    def get_number_of_g4(self):
        n2, n4 = NNcpp.GetNSyms(self._SymFunc)
        return n4

    def load_from_cfg(self, fname):
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
        NNcpp.LoadSymFuncFromCFG(self._SymFunc, fname)

    def save_from_cfg(self, fname):
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

    def get_symmetric_functions(self, atoms, Nx = 1, Ny = 1, Nz = 1):
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