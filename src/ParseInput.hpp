#ifndef PARSE_INPUT
#define PARSE_INPUT
#include "AtomicNetwork.hpp"

// Define the namespace of the configuration file
#define MODE "mode"


// Allowed modes are
#define M_GENERATE "generate"
#define M_LOAD "load"
#define M_TEST "test"
#define M_TEST_NEURON "neuro-test"

#define NETWORK_ENVIRONMENT "AtomicNetwork"

// If the mode is generate, you must insert the parameter of the NN
#define NLIM "n_lim"
// All the other environments are Symmetric functions and ensemble
#define NHIDDEN "n_hidden"
#define NPERLAYER "nodes_per_layer"
#define SAVE_PREFIX "save_prefix"
#define NX "N_sup_x"
#define NY "N_sup_y"
#define NZ "N_sup_z"

// All the keywords in the test mode
#define LOADNETWORK "load_nn_prefix"
#define CONFIG_FILE "test_config"
#define ATM_INDEX "atom_index"
#define ATM_COORD "atom_coord"
#define TEST_KIND "neuro_kind"
#define TEST_SINAPSIS "sinapsis"
#define TEST_BIAS "bias"
#define TEST_INDEX "neuro_index"
#define TEST_TYPE "neuro_type"
#define STEP_SIZE "step_size"
#define N_STEPS "n_steps"


/*
 * This function parses all the input file
 * and starts the appropriate calculation.
 */
void ParseInput(const char * inputfile) ;


/*
 * This function geneates the network from the input.
 */
void GetNetworkFromInput(const char * inputfile, AtomicNetwork * &network) ;
#endif