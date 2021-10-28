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

        

