import minnie, minnie.Ensemble as ENS
import minnie, minnie.SymmetricFunctions as SF
import minnie, minnie.AtomicNetwork as ANN

#import sscha, sscha.Ensemble
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def test_energy_forces(verbose = False):
    ENSEMBLE_LOC = "../../ReadEnsemble/new_ensemble"

    ensemble = ENS.Ensemble()
    ensemble.load_from_directory(ENSEMBLE_LOC,
                                 n_configs = 100,
                                 n_atoms = 40)

    symm_funcs = SF.SymmetricFunctions()
    symm_funcs.set_cutoff_radius(8) # 8 Angstrom cutoff
    symm_funcs.set_cutoff_function_type(1) # Flat derivative around the cutoff
    
    symm_funcs.add_g2_grid(r_min = 1.6, r_max = 6.5, N = 5)

    # Lets build an atomic neural network with this set of parameters
    network = ANN.AtomicNetwork()
    network.create_network_from_ensemble(symm_funcs, ensemble, pca_limit = 1500, hidden_layers_nodes = [10, 10])

    network.save_cfg("my_network.cfg")

    
    energy, force = network.get_energy( ensemble.get_configuration(3), compute_forces = True)

    if verbose:
        print("My energy:", energy)
        print("My force:", force)

    # Test for energy / forces gradients
    ID_CONFIG = 3
    ID_ATM = 0
    X_COORD = 0
    DX = 0.01
    NX = 20

    cfg = ensemble.get_configuration(ID_CONFIG)
    coords, types, uc = cfg.get_coords_types_uc()

    energies = []
    ff = []
    xx = []

    for i in range(NX):
        x = coords[ID_ATM, X_COORD] + DX * i
        new_coords = coords.copy()
        new_coords[ID_ATM, X_COORD] = x

        cfg.set_coords_types(new_coords, types, uc)
        en, f = network.get_energy(cfg, True)

        energies.append(energy)
        ff.append(f[ID_ATM, X_COORD])
        xx.append(x)

    other_ff = -np.gradient(energies, xx)
    print (np.diff(energies))


    if verbose:
        plt.plot(xx, ff, label = "NN Forces")
        plt.plot(xx, other_ff, ls = "--", label = "Numeric derivatives")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("F")
        plt.tight_layout()
        plt.show()
    
    assert np.max(np.abs(other_ff[1:-1] - ff[1:-1])) / np.mean(np.abs(ff)) < 1e-6
    
        

    
    

if __name__ == "__main__":
    test_energy_forces(True)
