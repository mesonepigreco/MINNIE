#include <iostream>
#include <string>
#include <string>
#include <libconfig.h++>
#include <stdexcept>


#include "symmetric_functions.hpp"
#include "atoms.hpp"

using namespace std;
using namespace libconfig;

void PrintUsage(void);

int main(int argc, char * argv[]) {
    // Load the configuration

    // Check if one argument has been passed
    if (argc != 7) {
        PrintUsage();
        cerr << "Error, you must specify 6 arguments." << endl;
        exit(EXIT_FAILURE);
    }

    int atm_index, cart_dir;
    int N_steps;
    double dx;

    try {
        atm_index = atoi(argv[1]);
        cart_dir = atoi(argv[2]);
        N_steps = atoi(argv[3]);
        dx = atof(argv[4]);
    } catch (...) {
        cerr << "Error while parsing the input." << endl;
        cerr << "Please, be shure that they match with the description." << endl;
        throw;
    }

    cout << "# atm_index = " << atm_index << endl;
    cout << "# cart_dir = " << cart_dir << endl;
    cout << "# N_steps = " << N_steps << endl;
    cout << "# dx = " << dx << endl;


    int N_syms, nat;
    
    // Setup the symmetric functions
    SymmetricFunctions * sym_func = new SymmetricFunctions();
    sym_func->LoadFromCFG(argv[5]);

    // Get the atomic configuration
    Atoms * atoms = new Atoms(argv[6]);

    N_syms = sym_func->GetTotalNSym(atoms->GetNTypes());
    nat = atoms->GetNAtoms();

    cout << "# N_sym = " << N_syms << endl;

    // Check if the input atomic type and coord are good
    if (atm_index < 0 || atm_index >= atoms->GetNAtoms()) {
        cerr << "Error, the atomic index is out of the allowed range" << endl;
        throw invalid_argument("atm_index must be between 0 and " + to_string(atoms->GetNAtoms()-1));
    }
    if (cart_dir < 0 || cart_dir >= 3) {
        cerr << "Error, the cartesian index is out of the allowed range" << endl;
        throw invalid_argument("cart_dir must be between 0 and 2");
    }
    if (N_steps <= 0) 
        throw invalid_argument("N_steps must be positive");

    double * sym_values = new double[N_syms*nat];
    double * sym_deriv = new double[N_syms*nat];

    cout << "# Step; S1; DS1/DX; S2; DS2/DX ..." << endl;
    // Start moving the coordinate
    for (int i = 0; i < N_steps; ++i) {
        // Get the symmetric functions
        sym_func->GetSymmetricFunctions(atoms, 1, 1, 1, sym_values);

        // For each symmetric function get the derivative with respect of the atom
        sym_func->GetDerivatives(atoms, 1, 1, 1, atm_index, cart_dir, sym_deriv);

        // Print on stdout
        cout << scientific << i * dx;
        for (int j = 0; j < N_syms * nat; ++j) {
            cout << "\t"<<  sym_values[j] << "\t" << sym_deriv[j];
        }
        cout << endl;

        // Move the atom
        atoms->coords[3 * atm_index + cart_dir] += dx;
    }

    return EXIT_SUCCESS;
}

void PrintUsage(void) {
    cout << "TEST SYMMETRY DERIVATIVES" << endl;
    cout << "=========================" << endl;
    cout << endl;
    cout << "This program shift one atom and prints symmetric functions and the derivatives." << endl;

    cout << "Usage:" << endl;
    cout << "./test_ensemble.exe <atm_id> <coord> <N_steps> <dx> <sym_func> <atm_conf>" << endl;
    cout << endl;
    cout << "atm_id    :  The atomic index to be moved" << endl;
    cout << "coord     :  The cartesian coordinate (0:x, 1:y, 2:z)" << endl;
    cout << "N_steps   :  The number of total steps" << endl;
    cout << "dx        :  The step size." << endl;
    cout << "sym_func  :  The .cfg file containing the symmetric functions." << endl;
    cout << "atm_conf  :  The .scf file containing the atomic structure." << endl;
    cout << endl;
    cout << endl;
}