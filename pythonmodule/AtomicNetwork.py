from __future__ import print_function
from __future__ import division

import numpy as np
import minnie, minnie.SymmetricFunctions as SF

import NNcpp

class AtomicNetwork:

    def __init__(self, filename = None):
        """
        INIT THE ATOMIC NEURAL NETWORK
        ==============================

        This function creates the atomic neural network.
        It accepts an optional variable that specifies the file name of the network.

        Parameters
        ---------- 
            filename : string
                The configuration file from which you want to load the atomic network.
                If missing, the network is not initialized
        """

        self._minnie = None 
        if filename:
            self.load_from_cfg(filename)

    def is_initialized(self):
        """
        Check if it has been initialized
        """
        if self._minnie:
            return False
        return True

    def create_network_from_ensemble(self, symmetryc_functions, ensemble, pca_limit, hidden_layers_nodes):
        """
        CREATE A NEW ATOMIC NETWORK
        ===========================

        Generate the atomic network.
        This function employs the ensemble to perform the PCA analysis of the chosen symmetric functions.
        The PCA is a way of performing a rotations of the symmetric functions to clean linear correlations between them. 
        It also automatically divide the first input layer for the variance.
        Thus the ensemble must be representative of the configurations on which the NN will be trained.

        Parameters
        ----------
            symmetric_functions : minnie.SymmetricFunctions.SymmetricFunctions
                The symmetric functions to describe the ensemble.
            ensemble : minnie.Ensemble.Ensemble
                The ensemble on which the PCA is performed.
            pca_limit : int
                How many PCA descriptors to emply as first layer of the network.
            hidden_layers_nodes : list of int
                How many nodes in each hidden layers.

        """
        if self._minnie:
            del self._minnie

        n_hidden = len(hidden_layers_nodes)
        hidlayer = np.zeros(n_hidden, dtype = np.intc)
        hidlayer[:] = hidden_layers_nodes

        assert all(hidlayer > 0), "Error, the number of nodes for hidden layer must be strictly positive"

        self._minnie = NNcpp.CreateAtomicNN(symmetryc_functions, ensemble, pca_limit, n_hidden, hidlayer)

    def load_from_cfg(self, filename):
        """
        Load the neural network from the specified file.
        Note: this overrides the original neural network.
        """

        if self._minnie:
            del self._minnie


        self._minnie = NNcpp.LoadNNFromCFG(filename)

    def save_to_cfg(self, filename):
        """
        Save the nn to the configuration file specified.
        """

        if not self.is_initialized():
            raise ValueError("Error, the NN must be initialized.")

        
        NNcpp.SaveNNToCFG(self._minnie, filename)

        

