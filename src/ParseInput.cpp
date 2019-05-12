#include <iostream>
#include <string>

#include <libconfig.h++>

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


using namespace std;
using namespace libconfig;




void ParseInput(const char * inputfile) {

    Config cfg;

    // Try to open the configuration file
    try {
        cfg.readFile(inputfile);
    } catch (const FileIOException &e) {
        cerr << "Error while opening " << inputfile << endl;
        throw;
    } catch (const ParseException &e) {
        cerr << "Syntax error in input file " << inputfile << endl;
        cerr << "Line: " << e.getLine() << endl;
        cerr << e.getError() << endl;
        throw;
    }

    // Get the mode
    const Setting& root = cfg.getRoot();
    string mode;
    try {
        if (!root.lookupValue(MODE, mode)) {
            cerr << "Error, keyword " << MODE << " is required!" << endl;
            cerr << "Aborting " << endl;
            exit(EXIT_FAILURE);
        }
    } catch (const SettingTypeException &e) {
        cerr << "Error, key " << MODE << " must be a string" << endl;
        cerr << e.getPath() << endl;
        throw;
    } catch (const SettingException & e) {
        cerr << "Error with " << e.getPath() << endl;
        cerr << "Please check carrefully!" << endl;
        throw;
    }

    
    int Nx, Ny, Nz;
    try {
        Nx = root.lookup(NX);
        Ny = root.lookup(NY);
        Nz = root.lookup(NZ);
    } catch (SettingNotFoundException& e) {
        cerr << "Error, setting " << e.getPath() << " not found." << endl;
        throw;
    } catch (SettingTypeException& e) {
        cerr << "Error, setting " << e.getPath() << " has the wrong type." << endl;
        cerr << "       required an integer." << endl;
        throw;
    } catch (...) {
        cerr << "Generic error while reading the supercell from the input file." << endl;
        cerr << "Please, check more carefully your input." << endl;
        throw;
    }

    // Parse the mode
    if (mode == M_GENERATE) {
        cout << "Using mode " << M_GENERATE << "..." << endl;

        // Read the network parameters
        int Nlim, Nhidden, Nnodes;
        //int Natoms, Nconfigs;
        string save_prefix, ens_folder;
        try {
            Nlim = root.lookup(NLIM);
            Nhidden = root.lookup(NHIDDEN);
            Nnodes = root.lookup(NPERLAYER);
            /* Natoms = root.lookup(ENSEMBLE_NATOMS);
            Nconfigs = root.lookup(ENSEMBLE_NCONFIG); */

            if (!root.lookupValue(SAVE_PREFIX, save_prefix)) {
                cerr << "Error, please provide a saving path thorugh " << SAVE_PREFIX << endl;
                exit(EXIT_FAILURE);
            }
        } catch (const SettingNotFoundException &e) {
            cerr << "Error, missing one of the required keywords:" << endl;
            cerr << e.getPath() << endl;
            throw;
        } catch (const SettingTypeException &e) {
            cerr << "Error, setting " << e.getPath() << " is of the wrong type." << endl;
            throw;
        } catch (const SettingException &e) {
            cerr << "Error with keyword:" << e.getPath() << endl;
            cerr << "Please check carrefully." << endl;
            throw;
        }

        cout << "Nlim:" << Nlim << endl;

        // Load the symmetric functions 
        SymmetricFunctions* symf = new SymmetricFunctions();
        try {
            symf->LoadFromCFG(inputfile);
        } catch (...) {
            cerr << "Error, you need to specify a valid environment for symmetric functions" << endl;
            throw;
        }

        // Load the ensemble
        Ensemble * ensemble = new Ensemble();
        try {
            ensemble->LoadFromCFG(inputfile);
        } catch (...) {
            cerr << "Error, you need to specify a valid environment for the ensemble" << endl;
            throw;
        }

        cout << "Building the atomic network..." << endl;
        AtomicNetwork * atm = new AtomicNetwork(symf, ensemble, Nlim, Nhidden, NULL, Nnodes);
        cout << "Saving the network into " << save_prefix << endl;
        atm->SaveCFG(save_prefix.c_str());
    } else if (mode == M_TEST || mode == M_TEST_NEURON) {
        // Perform the test on the neural network

        // Get the wanted network
        string network_prefix;
        string configuration_fname;
        int atom_index = 0, atom_coord= 0, n_steps = 10;
        double step_size = 1e-2;
        bool test_sinapsis = false;
        int neuro_index = 0;
        int type_index = 0;
        string test_kind;
        try {
            if (!root.lookupValue(LOADNETWORK, network_prefix)) {
                cerr << "Error, within mode " << M_TEST << endl;
                cerr << "       you need to provide the key: " << LOADNETWORK << endl;
                exit(EXIT_FAILURE);
            }
            if (!root.lookupValue(CONFIG_FILE, configuration_fname) && mode == M_TEST) {
                cerr << "Error, you must provide the testing configuration as: " << CONFIG_FILE << endl;
                exit(EXIT_FAILURE);
            }

            // Lookup all the other values
            if (mode == M_TEST) {
                root.lookupValue(ATM_INDEX, atom_index);
                root.lookupValue(ATM_COORD, atom_coord);
            } else {
                if (root.lookupValue(TEST_KIND, test_kind)) {
                    if (test_kind == TEST_BIAS) {
                        test_sinapsis = false;
                    } else if (test_kind == TEST_SINAPSIS){
                        test_sinapsis = true;
                    } else {
                        cerr << "Error, " << TEST_KIND << " argument can be only " << TEST_BIAS<< " or " << TEST_SINAPSIS << endl;
                        throw "";
                    }
                }

                neuro_index = root.lookup(TEST_INDEX);
                root.lookupValue(TEST_TYPE, type_index);
            }


            root.lookupValue(N_STEPS, n_steps);
            root.lookupValue(STEP_SIZE, step_size);

            if (atom_coord <0 || atom_coord >2 ) {
                cerr << "Error, the " << ATM_COORD << " keyword must be between [0, 2]" << endl;
                exit(EXIT_FAILURE);
            }
            if (n_steps <1) {
                cerr << "Error, " << N_STEPS << " must be at least 1" << endl;
                exit(EXIT_FAILURE);
            }
        } catch (SettingNameException& e) {
            cerr << "Error, wrong name while reading " << inputfile << endl;
            cerr << "Error in option: " << e.getPath() << endl;
            throw;
        } catch (SettingTypeException &e) {
            cerr << "Error in file " << inputfile << endl;
            cerr << "Wrong type of the option:" << e.getPath() << endl;
            throw;
        } catch (SettingNotFoundException &e) {
            cerr << "Error while reading the file " << inputfile << endl;
            cerr << "Required setting: " << e.getPath() << " not found!" <<endl;
            cerr << "Please, check carefully your input." << endl;
            cerr << e.what() << endl;
            throw;
        } catch (...) {
            cerr << "Generic error while reading the file " << inputfile << endl;
            cerr << "FILE: " << __FILE__ << "LINE: " << __LINE__ << endl;
            throw;
        }



        // Load the atomic neural network
        AtomicNetwork * atomic_network = new AtomicNetwork(network_prefix.c_str());

        // Load the given configuration or ensemble
        Atoms * config;
        Ensemble * ensemble;


        if (mode == M_TEST) 
            config = new Atoms(configuration_fname);
        else
        {
            ensemble = new Ensemble();
            ensemble->LoadFromCFG(inputfile);
        }
        


        // Check that the input is consistent with the given configuration
        double * forces;
        if (mode == M_TEST) {
            if (atom_index < 0 || atom_index >= config->GetNAtoms()) {
                cerr << "Error, the atomic index must be in the range [0, " << config->GetNAtoms() << ")" << endl;
                exit(EXIT_FAILURE);
            }

            forces = new double[config->GetNAtoms() * 3];
        }

        // Move the atom and get energy / force
        double energy, grad;



        double ** grad_biases = new double * [atomic_network->N_types];
        double ** grad_sinapsis = new double * [atomic_network->N_types];

        if (mode == M_TEST_NEURON) {
            for (int i = 0; i < atomic_network->N_types; ++i) {
                int n_biases = atomic_network->GetNNFromElement(atomic_network->atomic_types.at(i))->get_nbiases();
                int n_sinapsis = atomic_network->GetNNFromElement(atomic_network->atomic_types.at(i))->get_nsinapsis();

                
                grad_sinapsis[i] = new double[n_sinapsis];
                grad_biases[i] = new double[n_biases];
            }

            cout << "# Coord; Loss function; Gradient" << endl ;
        }
        else {
            cout << "# Coord; Energy; Force" << endl ;
        }

        for (int i = 0; i < n_steps ; ++i) {
            // Compute the atom energy and force
            if (mode == M_TEST) {
                energy = atomic_network->GetEnergy(config, forces);
                // Print on output
                cout << std::scientific <<  i*step_size << "\t" <<  energy << "\t" << forces[3 * atom_index + atom_coord] << endl;

                // Move the atom for the next step
                config->coords[3 * atom_index + atom_coord] += step_size;
            }
            else {
                // Put the gradient to zero
                for (int j = 0; j < atomic_network->N_types; ++j) {
                    int n_biases = atomic_network->GetNNFromElement(atomic_network->atomic_types.at(j))->get_nbiases();
                    int n_sinapsis = atomic_network->GetNNFromElement(atomic_network->atomic_types.at(j))->get_nsinapsis();

                    for (int k = 0; k < n_biases; ++k) {
                        grad_biases[j][k] = 0;
                    }
                    for (int k = 0; k < n_sinapsis; ++k) {
                        grad_sinapsis[j][k] = 0;
                    }
                }

                // Compute the loss function
                energy = atomic_network->GetLossGradient(ensemble, 1, 0, grad_biases, grad_sinapsis);

                if (test_sinapsis) {
                    grad = grad_sinapsis[type_index][neuro_index];
                    atomic_network->GetNNFromElement(type_index)->update_sinapsis_value(neuro_index, step_size);
                } else {
                    grad = grad_biases[type_index][neuro_index];
                    atomic_network->GetNNFromElement(type_index)->update_biases_value(neuro_index, step_size);
                }
                cout << std::scientific <<  i*step_size << "\t" <<  energy << "\t" << grad << endl;
            }
        }

        // Destroy
        if (mode == M_TEST_NEURON) {
            delete ensemble;
            for (int i = 0; i < atomic_network->N_types; ++i) {
                delete[] grad_biases[i];
                delete[] grad_sinapsis[i];
            }
        } else {
            delete config;
            delete[] forces;
        }
        delete[] grad_biases;
        delete[] grad_sinapsis;

        delete atomic_network;
    } 
    else {
        cerr << "Error, the variable " << MODE << " can be only:" << endl;
        cerr << "    1) " << M_GENERATE << endl;
        cerr << "    2) " << M_TEST << endl;
        cerr << "    3) " << M_TEST_NEURON << endl;
        cerr << " The value '" << mode << "' is not allowed!" << endl;
        exit(EXIT_FAILURE);
    }
}




void GetNetworkFromInput(const char * inputfile, AtomicNetwork * &network) {

    Config cfg;

    // Try to open the configuration file
    try {
        cfg.readFile(inputfile);
    } catch (const FileIOException &e) {
        cerr << "Error while opening " << inputfile << endl;
        throw;
    } catch (const ParseException &e) {
        cerr << "Syntax error in input file " << inputfile << endl;
        cerr << "Line: " << e.getLine() << endl;
        cerr << e.getError() << endl;
        throw;
    }



    // Get the mode
    const Setting& root = cfg.getRoot();
    if (!root.exists(NETWORK_ENVIRONMENT)) {
        cerr << "Error, environment " << NETWORK_ENVIRONMENT << " is required." << endl;
        throw "";
    }
    const Setting& network_env = root[NETWORK_ENVIRONMENT];

    string mode;

    try {
        network_env.lookupValue(MODE, mode);
    } catch (const SettingTypeException &e) {
        cerr << "Error, wrong type for option " << e.getPath() << endl;
        cerr << e.what() << endl;
        throw;
    }

    // Parse the mode
    if (mode == M_GENERATE) {
        cout << "Using mode " << M_GENERATE << "..." << endl;

        // Read the network parameters
        int Nlim, Nhidden, Nnodes;
        //int Natoms, Nconfigs;
        string save_prefix, ens_folder;
        try {
            Nlim = network_env.lookup(NLIM);
            Nhidden = network_env.lookup(NHIDDEN);
            Nnodes = network_env.lookup(NPERLAYER);
            /* Natoms = network_env.lookup(ENSEMBLE_NATOMS);
            Nconfigs = network_env.lookup(ENSEMBLE_NCONFIG); */

            //network_env.lookupValue(SAVE_PREFIX, save_prefix);
        } catch (const SettingNotFoundException &e) {
            cerr << "Error, missing one of the required keywords:" << endl;
            cerr << e.getPath() << endl;
            throw;
        } catch (const SettingTypeException &e) {
            cerr << "Error, setting " << e.getPath() << " is of the wrong type." << endl;
            throw;
        } catch (const SettingException &e) {
            cerr << "Error with keyword:" << e.getPath() << endl;
            cerr << "Please check carrefully." << endl;
            throw;
        }

        cout << "Nlim:" << Nlim << endl;

        // Load the symmetric functions 
        SymmetricFunctions* symf = new SymmetricFunctions();
        try {
            symf->LoadFromCFG(inputfile);
        } catch (...) {
            cerr << "Error, you need to specify a valid environment for symmetric functions" << endl;
            throw;
        }

        // Load the ensemble
        Ensemble * ensemble = new Ensemble();
        try {
            ensemble->LoadFromCFG(inputfile);
        } catch (...) {
            cerr << "Error, you need to specify a valid environment for the ensemble" << endl;
            throw;
        }

        cout << "Building the atomic network..." << endl;
        network = new AtomicNetwork(symf, ensemble, Nlim, Nhidden, NULL, Nnodes);
    } else if (MODE == M_LOAD) {
        string network_prefix;
        try {
            if (!root.lookupValue(LOADNETWORK, network_prefix)) {
                cerr << "Error, within mode " << M_TEST << endl;
                cerr << "       you need to provide the key: " << LOADNETWORK << endl;
                exit(EXIT_FAILURE);
            }
        } catch (SettingTypeException &e) {
            cerr << "Error in file " << inputfile << endl;
            cerr << "Wrong type of the option:" << e.getPath() << endl;
            throw;
        } catch (SettingNotFoundException &e) {
            cerr << "Error while reading the file " << inputfile << endl;
            cerr << "Required setting: " << e.getPath() << " not found!" <<endl;
            cerr << "Please, check carefully your input." << endl;
            cerr << e.what() << endl;
            throw;
        } catch (...) {
            cerr << "Generic error while reading the file " << inputfile << endl;
            cerr << "FILE: " << __FILE__ << "LINE: " << __LINE__ << endl;
            throw;
        }

        network = new AtomicNetwork(network_prefix.c_str());
    } else {
        cerr << "Error, the parameter: " << MODE << " given (" << mode.c_str() << ") is not supported." << endl;
        throw "";
    }
}