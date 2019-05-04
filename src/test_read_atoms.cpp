/*
 * In this simple example we test the ability of the code to read an atomic structure
 * We  read a standard scf file written in angstrom.
 * 
 * This program takes as input the scf file of the atoms, and it will print on
 * stdout the atomic coordinates.
 * 
 * If two other parameters are specified, then a supercell is generated.
 */

#include <iostream>
#include <fstream>
#include <string>
#include "atoms.hpp"

using namespace std;

void PrintUsage();

int main(int argn, char * argv[]){
    // Check if two arguments are provided
    cout << " TEST THE ATOMIC READ " << endl;
    cout << " ==================== " << endl << endl;
    if (argn != 3 && argn != 6) {
        PrintUsage();
        cerr << "ERROR: please provide a valid scf file." << endl;
        cout << "Aborting." << endl;
        return EXIT_FAILURE;
    }

    // Get the number of atoms
    int nat;
    string path;

    // Check if the first is a correct nuber
    try {
        nat = atoi(argv[1]);
        if (nat <= 0) {
            throw "Error, NAT must be > 0";
        }
    } catch (...) {
        cerr << "Error, the first argument must be a valid int." << endl;
        throw;
    }

    // Get the path
    path.assign(argv[2]);

    int Nx = 1, Ny = 1, Nz = 1;
    // Check if the size of the cell is provided
    if (argn == 6) {
        try {
            Nx = atoi(argv[3]);
            Ny = atoi(argv[4]);
            Nz = atoi(argv[5]);

            if (Nx <= 0 || Ny <= 0 || Nz <= 0)
                throw invalid_argument("invalid cell size");
        } catch (...) {
            cerr << "Error, the supercell must be a valid integer" << endl;
            throw;
        }
    }

    // Parse the atom object
    Atoms structure(nat);
    structure.ReadSCF(path);

    // Print the coordinates
    cout << "Atom read!" << endl;

    // generate the supercell
    Atoms *supercell;
    structure.GenerateSupercell(Nx, Ny, Nz, supercell);

    // Print the unit cell
    cout << endl;
    cout << "UNIT CELL:" << endl;
    for (int i = 0; i < 3; ++i) {
        cout << structure.unit_cell[3*i] << "  " << structure.unit_cell[3*i + 1] << "  " <<  structure.unit_cell[3*i + 2] << endl;
    }
    cout << endl;
    cout << "Supercell size:" << Nx << " " << Ny << " " << Nz << endl;
    cout << "Supercell N_atoms: " << supercell->GetNAtoms() << endl;
    cout << "Supercell Coordinates:" << endl;
    double x, y, z;
    for (int i = 0; i < supercell->GetNAtoms(); ++i) {
        supercell->GetAtomsCoords(i, x, y, z);
        cout << supercell->types[i] << '\t' << x << '\t' << y << '\t' << z << endl;
    }
    cout << endl;
    cout << "Done!" << endl;

    return 0;
}


void PrintUsage() {
    cout << "USAGE:" << endl;
    cout << endl;
    cout << "./test_read_atoms.x NAT FNAME [NX NY NZ]";
    cout << endl;
    cout << "NAT    \t:\tNumber of atoms        " << endl;
    cout << "FNAME  \t:\tScf file path          " << endl;
    cout << "[NX]   \t:\tThe supercell size on x" << endl;
    cout << "[NY]   \t:\tThe supercell size on y" << endl;
    cout << "[NZ]   \t:\tThe supercell size on z" << endl;
    cout << endl;
    cout << "Arguments marked with square brakets [x] are optional." << endl;
    cout << endl;
}