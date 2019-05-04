/*
 * This header contains the directive to build a neural network for atoms
 * It allows to be used toghether with atoms with different configurations.
 * 
 * This is a must to have.
 */

#ifndef HEADER_ATOMICNETWORK
#define HEADER_ATOMICNETWORK
#include <algorithm>
#include "network.hpp"
#include <string>
#include "symmetric_functions.hpp"
#include <libconfig.h++>
#include "ensemble.hpp"
#include "analyze_ensemble.hpp"
#include "matrix.h"

// Define here the keyword for saving
#define AN_PCA "PCA"
#define AN_NLIM "N_lim"
#define AN_NSYMS "N_syms"
#define AN_EIGVECT "eigenvector_"
#define AN_EIGVALS "variances"
#define AN_NTYPES "n_types"
#define AN_ENVIRON "AtomicNetwork"
#define AN_TYPELIST "types"

class AtomicNetwork {
private:
    vector<NeuralNetwork *> atomic_network;
    vector<int> atomic_types;
    int N_types;
    // Symmetric functions
    SymmetricFunctions * symm_f;

    // The supercell for the symmetric function calculation
    //int Nx, Ny, Nz;

    // PCA variables on the symmetric functions
    double * eigvals, *eigvects;
    int N_lim;


public:
    /*
     * Constructor method.
     */
    // Construct an atomic network by loading one from file
    AtomicNetwork(const char * PREFIX);

    // Construct an empty network by standard initialization
    /*
     * This will build an atomic network from the external values
     * The PCA will be performed on the provided ensemble, and only the
     * highest N_lim linear combinations of the symmetric functions will be used.
     * It is suggested to check the reuslt of the PCA with the analyze_ensemble utility
     * provided with this code to see if some symmetric function can be easily removed.
     * 
     * You need to provide also the number of hidden layers for each atomic network 
     * and the number of nodes for each layer.
     * 
     * Parameters
     * ----------
     *      symmetric_functions : 
     *          The Class of the symmetric functions to be used as input
     *      ensemble :
     *          The atomic ensemble to be used to initialize the network.
     *      N_lim :
     *          The number of symmetric functions to be used.
     *      N_hidden:
     *          How many hidden layers?
     *      nodes_per_hidden_layer: 
     *          How many nodes per layer? Leave NULL if you want a fixed number for all.
     *      nodes_hl :
     *          Home many nodes per layer? Leave 0 if you want a different number for each layer.
     */
    AtomicNetwork(SymmetricFunctions*, Ensemble*, int Nx, int Ny, int Nz, int N_lim, int N_hidden,
        int * nodes_per_hidden_layer = NULL, int nodes_hl = 0);
    ~AtomicNetwork();

    /*
     * Use the current deep neural network to get the atomic energies.
     * 
     * This function can also be used to compute the gradient on the network and
     * the force on the atomic configuration, if the corresponding passed pointers are not NULL.
     * 
     * Nx, Ny and Nz are the supercell.
     * 
     * The double pointer refers to an array of gradients, one per each atomic neural network.
     */
    double GetEnergy(Atoms * coords, double * forces = NULL, int Nx = 1, int Ny = 1, int Nz = 1, double ** grad_biases = NULL, double ** grad_sinapsis = NULL);


    /*
     * Save/Load the current network in a cfg file.
     * You need to provide the prefix to the network. 
     * The result will be:
     * 
     * PREFIX_main.cfg : This file contains the info about the symmetry functions,
     *                   the PCA, and the atomic types that can be found in the network.
     * PREFIX_YYY.cfg  : The YYY are the atomic types, it contains the info about the single atomic network.
     */
    void SaveCFG(const char * PREFIX);
    void LoadCFG(const char * PREFIX);

    NeuralNetwork * GetNNFromElement(int element_type);

};

#endif