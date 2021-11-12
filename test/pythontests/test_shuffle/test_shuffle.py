import sys, os

import minnie, minnie.Ensemble as ENS
import minnie.SymmetricFunctions as SF
import minnie.AtomicNetwork as ANN
import minnie.Atoms as ATM

#import sscha, sscha.Ensemble
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def test_shuffle(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    ENSEMBLE_LOC = "../../ReadEnsemble/new_ensemble"

    
    ensemble = ENS.Ensemble()
    ensemble.load_from_directory(ENSEMBLE_LOC,
                                 n_configs = 5,
                                 n_atoms = 40)


    cfgs = [ensemble.get_configuration(x) for x in range(ensemble.get_n_configs())]
    energy_forces = [ensemble.get_energy_forces(x) for x in range(ensemble.get_n_configs())]
    ensemble.shuffle()

    indices = []
    for i in range(ensemble.get_n_configs()):
        cfg = ensemble.get_configuration(i)
        en, f = ensemble.get_energy_forces(i)

        index = 0
        for j in range(ensemble.get_n_configs()):
            if en == energy_forces[j][0]:
                index = j
                break
        print("INDEX:", index)
        
        assert np.max(np.abs(f - energy_forces[index][1])) < 1e-8

        coords, _, _ = cfg.get_coords_types_uc()
        c2, _, _ = cfgs[index].get_coords_types_uc()
        assert np.max(np.abs(c2 - coords)) < 1e-8
        indices.append(index)
    if verbose:
        print (indices)
            

    
    
    

if __name__ == "__main__":
    test_shuffle(True)
