from __future__ import print_function

import numpy as np


import minnie, minnie.Atoms as ATM
import NNcpp

# Override the print function
from minnie.Parallel import pprint as print

class Ensemble:

    def __init__(self):
        """
        Create an empty ensemble
        """
        
        self._ensemble = None

    def initialize(self, n_structures, n_atoms):
        """
        Initialize the ensemble given the number of structures and atoms.
        Note: this will override any previous declaration of the ensemble.
        """
        if self._ensemble:
            del self._ensemble

        assert n_structures > 0
        assert n_atoms > 0


        self._ensemble = NNcpp.CreateEnsembleClass(n_structures, n_atoms)

    def load_from_directory(self, path_to_dir, n_configs, n_atoms, population_id = 1, alat = 1., append = False):
        """
        Load the ensemble from the directory.
        The data inside the directory are the same as those saved by the python-sscha ensemble (not binary)

        Parameters
        ----------
            path_to_dir : string
                The path to the location of the ensemble
            n_configs : int
                The number of configurations in the ensemble
            n_atoms : int
                The number of atoms in each configuration
            population_id : int
                The id of the population (if more ensemble are in the same directory)
            alat : double
                The unit in Angstrom on which the configuration coordinates are specified.
                If not given, the coordinates are supposed to be measured in Angstrom.
            append : bool
                If True, the configurations are appended on the back of the ensemble.
        """

        if not self._ensemble:
            append = False
            self.initialize(n_configs, n_atoms)

        override = not append

        NNcpp.LoadEnsemble(self._ensemble, path_to_dir, n_configs, population_id, n_atoms, alat, override)

    def get_n_configs(self):
        """
        Returns the number of configurations in the ensemble
        """

        return NNcpp.GetNConfigsEnsemble(self._ensemble)

    def get_n_types(self):
        """
        Get the total number of different atomic species inside the ensemble.
        """

        return NNcpp.GetNTypsEnsemble(self._ensemble)

    def get_configuration(self, index):
        """
        Get the specified configuration from the ensemble.

        Parameters
        ----------
            index : int
                The index of the configuration

        Results
        -------
            config : minnie.Atoms.Atoms
                The configuration
        """

        _cfg = NNcpp.GetEnsembleConfig(self._ensemble, index)
        configuration = ATM.Atoms(0, atom_class = _cfg)

        return configuration


    def add_structures(self, list_of_atoms, energies = None, forces = None, stresses = None):
        """
        Initialize the ensemble with the list of atoms
        """

        if not self._ensemble:
            raise ValueError("Error, the ensemble must be initialized with the initialize method.")
        
        for i, atm in enumerate(list_of_atoms):
            energy = 0
            if energies:
                energy = energies[i]
            force = np.zeros( 3 * atm.N_atoms, dtype= np.double)
            if forces:
                force[:] = forces[i]
            stress = np.zeros(6, dtype = np.double)
            if stresses:
                stress[:] = stresses[i]

            NNcpp.OvverrideEnsembleIndex(i, self._ensemble, atm, energy, force, stress)
            
    
    def get_energy_forces(self, index):
        """
        GET ENERGY AND FORCES FOR THE GIVEN CONFIGURATION
        =================================================

        Get the energy and the forces for the index of the given configuration

        Parameters
        ----------
            index : int
                The index of the configuration.

        Results
        -------
            energy : float
                The value of the energy
            forces : ndarray( size = (n_atoms, 3), dtype = np.double, order = "C")
                The forces for each atom in the configuration
        """

        cfg = self.get_configuration(index)
        forces = np.zeros( (cfg.N_atoms, 3), dtype = np.double, order = "C")

        energy = NNcpp.Ensemble_GetEnergyForces(self._ensemble, forces, index, cfg.N_atoms)
        return energy, forces

    def convert_from_sscha(self, sscha_ensemble):
        """
        Fill the ensemble using a sscha ensemble.
        """

        new_atms = []
        forces = []
        stresses = []
        for i, s in enumerate(sscha_ensemble.structures):
            atm = ATM.Atoms(s.N_atoms)
            atm.set_from_cellconstructor(s)

            new_atms.append(atm)
            forces.append(sscha_ensemble.forces[i, :])
            stress = np.zeros(6, dtype = np.double)
            stress[:3] = np.diag(sscha_ensemble.streses[i, :, :])
            stress[3] =  sscha_ensemble.streses[i, 0,1]
            stress[4] =  sscha_ensemble.streses[i, 1,2]
            stress[5] =  sscha_ensemble.streses[i, 0,2]
            stresses.append(stress)
        
        self.add_structures(new_atms, sscha_ensemble.energies, forces, stresses)

    def get_list_of_atoms(self):
        """
        This method returns a list of the structures of the ensemble.
        All of them are minnie.Atoms.Atoms
        """
        return [self.get_configuration(i) for i in range( self.get_n_configs())]

    def shuffle(self):
        """
        RANDOM SHUFFLE
        ==============

        Random shuffle the order of the configurations inside the ensemble. 
        Usefull for the bootstrap procedure.
        """

        NNcpp.Ensemble_Shuffle(self._ensemble)

        
