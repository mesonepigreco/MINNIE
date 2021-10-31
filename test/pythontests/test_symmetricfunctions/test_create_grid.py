import minnie, minnie.Ensemble as ENS
import minnie, minnie.SymmetricFunctions as SF
#import sscha, sscha.Ensemble
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def test_grid_g2(verbose = False):
    ENSEMBLE_LOC = "../../ReadEnsemble/new_ensemble"

    ensemble = ENS.Ensemble()
    ensemble.load_from_directory(ENSEMBLE_LOC,
                                 n_configs = 100,
                                 n_atoms = 40)

    symm_funcs = SF.SymmetricFunctions()
    symm_funcs.set_cutoff_radius(8) # 8 Angstrom cutoff
    symm_funcs.set_cutoff_function_type(1) # Flat derivative around the cutoff
    
    symm_funcs.add_g2_grid(r_min = 1.6, r_max = 6.5, N = 5)

    mean, cvar = symm_funcs.pca_analysis(ensemble)

    pearson = SF.covariance_to_correlation(cvar)

    print(pearson[0,2])
    print(cvar[0,2] / np.sqrt(cvar[0,0] * cvar[2,2]))
    print(cvar[0,2])
    print(cvar[0,0], cvar[2,2])

    if verbose:
        plt.figure()
        plt.imshow(pearson)
        plt.title("Pearson correlation.")
        plt.colorbar()

        plt.figure()
        plt.title("Covariance matrix (absolute values)")
        MIN = 1e-12
        cvar[np.abs(cvar) < MIN] = MIN
        plt.imshow(np.abs(cvar), norm=LogNorm(vmin=np.min(np.abs(cvar)), vmax=np.max(cvar)))
        plt.colorbar()
        plt.show()


        
def test_grid_g4(verbose = False):
    ENSEMBLE_LOC = "../../ReadEnsemble/new_ensemble"

    ensemble = ENS.Ensemble()
    ensemble.load_from_directory(ENSEMBLE_LOC,
                                 n_configs = 100,
                                 n_atoms = 40)

    symm_funcs = SF.SymmetricFunctions()
    symm_funcs.set_cutoff_radius(8) # 8 Angstrom cutoff
    symm_funcs.set_cutoff_function_type(1) # Flat derivative around the cutoff
    
    symm_funcs.add_g4_grid(r_value = 5, N = 6)
    symm_funcs.save_cfg("new_g4_grid.cfg")

    mean, cvar = symm_funcs.pca_analysis(ensemble)

    pearson = SF.covariance_to_correlation(cvar)

    print(pearson[0,2])
    print(cvar[0,2] / np.sqrt(cvar[0,0] * cvar[2,2]))
    print(cvar[0,2])
    print(cvar[0,0], cvar[2,2])

    if verbose:
        plt.figure()
        plt.imshow(pearson)
        plt.title("Pearson correlation.")
        plt.colorbar()

        plt.figure()
        plt.title("Covariance matrix (absolute values)")
        MIN = 1e-12
        cvar[np.abs(cvar) < MIN] = MIN
        plt.imshow(np.abs(cvar), norm=LogNorm(vmin=np.min(np.abs(cvar)), vmax=np.max(cvar)))
        plt.colorbar()
        plt.show()

        
if __name__ == "__main__":
    test_grid_g4(True)
        
    
