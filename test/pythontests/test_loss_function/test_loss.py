import sys, os

import minnie, minnie.Ensemble as ENS
import minnie.SymmetricFunctions as SF
import minnie.AtomicNetwork as ANN
import minnie.Atoms as ATM

#import sscha, sscha.Ensemble
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def test_loss(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    ENSEMBLE_LOC = "../../ReadEnsemble/new_ensemble"

    
    ensemble = ENS.Ensemble()
    ensemble.load_from_directory(ENSEMBLE_LOC,
                                 n_configs = 100,
                                 n_atoms = 40)

    # Create the symmetric functions
    symm_funcs = SF.SymmetricFunctions()
    symm_funcs.set_cutoff_radius(8) # 8 Angstrom cutoff
    symm_funcs.set_cutoff_function_type(1) # Flat derivative around the cutoff
    
    symm_funcs.add_g2_grid(r_min = 1.6, r_max = 6.5, N = 5)
    symm_funcs.add_g4_grid(r_value = 3, N = 6)


    # Create the network
    network = ANN.AtomicNetwork()
    network.create_network_from_ensemble(symm_funcs, ensemble, pca_limit = 10, hidden_layers_nodes = [10, 10])

    # Compute the Loss function
    loss, g_b, g_s = network.get_loss_function(ensemble, 1.0, 0.0)

    if verbose:
        print("The loss function is: {}".format(loss))

    biases, synapsis = network.get_biases_synapsis()

    N_CHANGE = 10
    TYPE_CHANGE = 0
    ID_BIAS = 16
    DX = 0.01

    xx = []
    ff = []
    losses = []
    for i in range(N_CHANGE):
        x = biases[TYPE_CHANGE, ID_BIAS] + DX * i
        new_biases = biases.copy()
        new_biases[TYPE_CHANGE, ID_BIAS] = x

        network.set_biases_synapsis(new_biases, synapsis)
        loss, grad_biases, grad_synapsis = network.get_loss_function(ensemble, 1.0, 0.0)
        
        xx.append(x)
        ff.append(grad_biases[TYPE_CHANGE, ID_BIAS])
        losses.append(loss)

    other_ff = np.gradient(losses, xx)
    

    if verbose:
        plt.plot(xx, ff, label = "Bias gradient")
        plt.plot(xx, other_ff, ls = "--", label = "Numeric derivatives")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("F")
        plt.tight_layout()
        plt.show()
    
    assert np.max(np.abs(other_ff[1:-1] - ff[1:-1])) / np.mean(np.abs(ff)) < 1e-4, np.abs(other_ff[1:-1] - ff[1:-1]) / np.mean(np.abs(ff))


def test_train(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    ENSEMBLE_LOC = "../../ReadEnsemble/new_ensemble"

    
    ensemble = ENS.Ensemble()
    ensemble.load_from_directory(ENSEMBLE_LOC,
                                 n_configs = 100,
                                 n_atoms = 40)

    # Create the symmetric functions
    symm_funcs = SF.SymmetricFunctions()
    symm_funcs.set_cutoff_radius(8) # 8 Angstrom cutoff
    symm_funcs.set_cutoff_function_type(1) # Flat derivative around the cutoff
    
    symm_funcs.add_g2_grid(r_min = 1.6, r_max = 6.5, N = 5)
    symm_funcs.add_g4_grid(r_value = 3, N = 6)


    # Create the network
    network = ANN.AtomicNetwork()
    network.create_network_from_ensemble(symm_funcs, ensemble, pca_limit = 10, hidden_layers_nodes = [10, 10])

    network.train(ensemble, 5, verbose = verbose)

    if verbose:
        network.save_cfg("trained")
    

if __name__ == "__main__":
    test_train(True)
    test_loss(True)
    
