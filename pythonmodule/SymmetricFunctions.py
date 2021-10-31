from __future__ import print_function
from __future__ import division

# Load the original CPP Module
import NNcpp 
import numpy as np

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

    def add_g2_grid(self, r_min, r_max, N):
        """
        CREATE A GRID OF SYMMETRIC FUNCTIONS (G2)
        =========================================

        This method automatically creates a grid of symmetric functions to describe completely the
        space of the parameters with just coordination numbers.

        It creates only G2 functions centered around zero.

        Parameters
        ----------
            r_min : float
                The minimum radius probed.
            r_max : float
                The maximum radius probed.
            N : int
                The number of symmetric functions
        """

        r_lims = np.linspace(r_min, r_max, N)
        etas = 1 / (2 * r_lims**2)

        for i, eta in enumerate(etas):
            self.add_g2_function(0, eta)


    def add_g4_grid(self, r_value, N):
        """
        CREATE A GRID OF SYMMETRIC FUNCTIONS (G4)
        =========================================

        This method automatically creates a grid of symmetric functions to describe completely the
        space of the parameters with angles triplets.

        Higher values of N means higher sensitivity on the angles.

        Parameters
        ----------
            r_value : float
                The value of the radius above which the atoms are neglected.
            N : int
                The number of symmetric functions. For each N, 2 symmetric functions are added.
                One with lambda = 1 and the other with lambda = -1.
        
        """

        zetas = 2**np.arange(N)
        eta = 1 / (2 * r_value**2)

        for i, zeta in enumerate(zetas):
            self.add_g4_function(eta, zeta, 1)
            self.add_g4_function(eta, zeta, -1)

    def get_total_number_functions(self, ntyps):
        """
        Get the total number of symmetric function given the atomic types
        """

        return NNcpp.GetNSyms(self._SymFunc, ntyps)[2]

    def get_cutoff_function(self):
        r"""
        Returns a tuple with the type (0 or 1) and the radius.

        Cutoff functions are those defined in the Beheler paper 10.1002/qua.24890:

        Above :math:`R_c` the cutoff function is zero, below it is a value specified by the type.

        if type is 0 the function is

        .. math ::

            \frac 12 \left[ \cos\left(\frac{\pi R}{R_c}\right) + 1\right]

        if type is 1, the function is

        .. math ::
            
            \tanh^3\left(1 - \frac{R}{R_c})


        Results
        -------
            type : int
                Either 0 or 1. The typology of the cutoff function employed
            cutoff_radius : float
                The :math:`R_c` value above which the symmetry function is truncated.
        """

        return NNcpp.GetCutoffTypeRadius(self._SymFunc)

    def set_cutoff_radius(self, value):
        """
        Setup the radius for the cutoff function.
        """
        NNcpp.SetCutoffRadius(self._SymFunc, np.double(value))

    def set_cutoff_function_type(self, type):
        r"""
        Setup the cutoff function type (either 0 or 1).
        The cutoff function is 0 above the cutoff radius and the value identified by 
        the function type below:

        if type is 0 the function is

        .. math ::

            \frac 12 \left[ \cos\left(\frac{\pi R}{R_c}\right) + 1\right]

        if type is 1, the function is

        .. math ::
            
            \tanh^3\left(1 - \frac{R}{R_c})
        """

        if type != 0 and type != 1:
            raise ValueError("Error, type can be only 0 or 1")
        
        NNcpp.SetCutoffType(self._SymFunc, np.intc(type))

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

        results = NNcpp.GetSymmetricFunctionParameters(self._SymFunc, index, isg2)    
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

        tipology = "g2"
        if not isg2:
            tipology = "g4"

        old_params = self.get_parameters(index, tipology)
        
        for key in params:
            if not key in old_params:
                raise ValueError("Error, key '{}' not supported. Allowed keys are {}".format(key, list(old_params)))
            
            old_params[key] = params[key]

        if isg2:
            p1 = np.double(old_params["g2_Rs"])
            p2 = np.double(old_params["g2_eta"])
            p3 = np.intc(0)#params["g2_eta"])
        else:
            p1 = np.double(old_params["g4_zeta"])
            p2 = np.double(old_params["g4_eta"])
            p3 = np.intc(old_params["g4_lambda"])
        
        NNcpp.SetSymmetricFunctionParameters(self._SymFunc, np.intc(index), isg2, p1, p2, p3)

    def print_info(self):
        """
        PRINT INFO
        ==========

        Print a report on stdout on all the symmetric functions initialized.
        """
        NNcpp.SymPrintInfo(self._SymFunc)

    def add_g2_function(self, Rs, eta):
        """
        Add a new symmetric function of type G2 (non angular)
        """

        NNcpp.AddSymG2(self._SymFunc, np.double(Rs), np.double(eta))

    def add_g4_function(self, eta, zeta, lambd):
        """
        Add a new symmetric function of type G4
        """

        NNcpp.AddSymG4(self._SymFunc, np.double(eta), np.double(zeta), np.int(lambd))

    
    def get_number_of_g2(self):
        n2, n4, _ = NNcpp.GetNSyms(self._SymFunc,0)
        return n2

    def get_number_of_g4(self):
        n2, n4, _ = NNcpp.GetNSyms(self._SymFunc,0)
        return n4

    def load_cfg(self, fname):
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

    def save_cfg(self, fname):
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

    def get_symmetric_functions(self, atoms, Nx = 3, Ny = 3, Nz = 3):
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
                If you do not want periodic boundary conditions, use 1 for each value

        Results
        -------
            sym_funcs : ndarray( size = (N_atoms, N_syms), dtype = np.double)
                The symmetric functions for each atoms of the system.
        """

        sym_funcs = NNcpp.GetSymmetricFunctions(self._SymFunc, atoms._atoms, Nx, Ny, Nz)
        return sym_funcs

    def pca_analysis(self, ensemble, Nx=3, Ny=3, Nz=3):
        """
        PERFORM THE PCA ANALYSIS ON THE GIVEN SYMMETRIC FUNCTIONS
        =========================================================

        The PCA analysis consists in computing the mean values of each symmetric function
        per atom inside the ensemble and their covariance matrix.
        
        This method only returns the covariance matrix and the mean, you need to diagonalize them
        to assess the PCA.

        It is always very usefull to run this check to assess if there are redundant symmetric functions

        Parameters
        ----------
            ensemble : minnie.Ensemble.Ensemble
                The ensemble on which the PCA is performed.
            Nx, Ny, Nz : int
                The periodicity of the cell. 1,1,1 if no periodic boundary conditions.

        Results
        -------
            mean : ndarray(size = nsym)
                For each symmetry function, the mean value it has in the ensemble.
            cvar_mat : ndarray(size = (nsym, nsym))
                The covariance matrix between symmetric functions on the ensemble.
        """

        ntyps = ensemble.get_n_types()
        nsyms = self.get_total_number_functions(ntyps)

        means = np.zeros(nsyms, dtype = np.double)
        cvar_mat = np.zeros( (nsyms, nsyms), dtype = np.double, order = "C")

        NNcpp.GetCovarianceMatrix(ensemble._ensemble, self._SymFunc, Nx, Ny, Nz, means, cvar_mat)
        return means, cvar_mat


def covariance_to_correlation(cvar_mat):
    """
    Get the Pearson correlation matrix from the covariance matrix

    Parameters
    ----------
        cvar_mat : ndarray(size = (N,N))
            The covariance matrix
    
    Returns
    -------
        pearson_corr_mat : ndarray(size = (N,N))
            The pearson correlation coefficient
    """

    sigmas = np.sqrt(np.diag(cvar_mat))
    return cvar_mat / np.outer(sigmas, sigmas)