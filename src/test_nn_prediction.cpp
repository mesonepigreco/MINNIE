#include <iostream>
#include <string>

#include <libconfig.h++>

#include "AtomicNetwork.hpp"

#define NARGS 2

using namespace std;
using namespace libconfig;

// Define the namespace of the configuration file
#define MODE "mode"

// Allowed modes are
#define M_GENERATE "generate"
#define M_TEST "test"

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
#define STEP_SIZE "step_size"
#define N_STEPS "n_steps"



// Print usage of the program
void PrintInfo(void);

int main(int argc, char * argv[]) {
    
    // Get as first parameter the input file
    if (argc != NARGS) {
        PrintInfo();
        cerr << "Error, you must provide one argument." << endl;
        cerr << "       that specifies the input file." << endl;
        exit(EXIT_FAILURE);
    }

    Config cfg;

    // Try to open the configuration file
    try {
        cfg.readFile(argv[1]);
    } catch (const FileIOException &e) {
        cerr << "Error while opening " << argv[1] << endl;
        throw;
    } catch (const ParseException &e) {
        cerr << "Syntax error in input file " << argv[1] << endl;
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
            symf->LoadFromCFG(argv[1]);
        } catch (...) {
            cerr << "Error, you need to specify a valid environment for symmetric functions" << endl;
            throw;
        }

        // Load the ensemble
        Ensemble * ensemble = new Ensemble();
        try {
            ensemble->LoadFromCFG(argv[1]);
        } catch (...) {
            cerr << "Error, you need to specify a valid environment for the ensemble" << endl;
            throw;
        }

        cout << "Building the atomic network..." << endl;
        AtomicNetwork * atm = new AtomicNetwork(symf, ensemble, Nx, Ny, Nz, Nlim, Nhidden, NULL, Nnodes);
        cout << "Saving the network into " << save_prefix << endl;
        atm->SaveCFG(save_prefix.c_str());
    } else if (mode == M_TEST) {
        // Perform the test on the neural network

        // Get the wanted network
        string network_prefix;
        string configuration_fname;
        int atom_index = 0, atom_coord= 0, n_steps = 10;
        double step_size = 1e-2;
        try {
            if (!root.lookupValue(LOADNETWORK, network_prefix)) {
                cerr << "Error, within mode " << M_TEST << endl;
                cerr << "       you need to provide the key: " << LOADNETWORK << endl;
                exit(EXIT_FAILURE);
            }
            if (!root.lookupValue(CONFIG_FILE, configuration_fname)) {
                cerr << "Error, you must provide the testing configuration as: " << CONFIG_FILE << endl;
                exit(EXIT_FAILURE);
            }

            // Lookup all the other values
            root.lookupValue(ATM_INDEX, atom_index);
            root.lookupValue(ATM_COORD, atom_coord);
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
            cerr << "Error, wrong name while reading " << argv[1] << endl;
            cerr << "Error in option: " << e.getPath() << endl;
            throw;
        } catch (SettingTypeException &e) {
            cerr << "Error in file " << argv[1] << endl;
            cerr << "Wrong type of the option:" << e.getPath() << endl;
            throw;
        } catch (...) {
            cerr << "Generic error while reading the file " << argv[1] << endl;
            cerr << "Please, check carefully your input." << endl;
            throw;
        }


        // Load the atomic neural network
        AtomicNetwork * atomic_network = new AtomicNetwork(network_prefix.c_str());

        // Load the given configuration
        Atoms * config = new Atoms(configuration_fname);

        // Check that the input is consistent with the given configuration
        if (atom_index < 0 || atom_index >= config->GetNAtoms()) {
            cerr << "Error, the atomic index must be in the range [0, " << config->GetNAtoms() << ")" << endl;
            exit(EXIT_FAILURE);
        }

        // Move the atom and get energy / force
        double energy;
        double * forces = new double[config->GetNAtoms() * 3];


        cout << "# Coord; Energy; Force" << endl ;
        for (int i = 0; i < n_steps ; ++i) {
            // Compute the atom energy and force
            energy = atomic_network->GetEnergy(config, forces);

            // Print on output
            cout << std::scientific <<  i*step_size << "\t" <<  energy << "\t" << forces[3 * atom_index + atom_coord] << endl;

            // Move the atom for the next step
            config->coords[3 * atom_index + atom_coord] += step_size;
        }

        // Destroy
        delete[] forces;
        delete atomic_network;
        delete config;
    } else {
        cerr << "Error, the variable " << MODE << " can be only:" << endl;
        cerr << "    1) " << M_GENERATE << endl;
        cerr << "    2) " << M_TEST << endl;
        cerr << " The value '" << mode << "' is not allowed!" << endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}


void PrintInfo(void) {
    cout << "TEST THE ATOMIC NETWORK" << endl;
    cout << "=======================" << endl << endl;
    cout << "Usage:" << endl;
    cout << "  test_nn_prediction.exe <input_file.cfg>" << endl << endl;
    cout << " - input_file.cfg   :  The input file that specify what to do." << endl;
    cout << endl;
}