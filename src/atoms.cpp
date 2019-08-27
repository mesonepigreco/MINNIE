#include <stdio.h>
#include <iostream>
#include <fstream>
#include "atoms.hpp"
#include <vector>

#define DEBUG_ATOMS 0

int GetNatmFromFile(string path) {
  ifstream input(path.c_str());

  if (!input) {
    cerr << "Error while reading " << path << endl;
    throw invalid_argument("File not found");
  }

  string line;
  int nat = 0;
  bool reading = false;
  while(getline(input, line)) {
    if (line.find("ATOMIC") != string::npos) {
      reading = true;
    } else {
      if (reading) {
        if (!line.empty())
          nat++;
      }
    }
  }
  input.close();
  return nat;
}

// Constructor of the class
Atoms::Atoms(int N) {
  N_atoms = N;
  N_types = 1;
  
  // Initialize the types and coords
  coords = (double*) malloc( sizeof(double) * 3 * N);
  types = (int*) calloc(sizeof(int), N);

}

Atoms::Atoms(string scf_file) {
  N_atoms = GetNatmFromFile(scf_file);
  N_types = 1;
  
  // Initialize the types and coords
  coords = (double*) malloc( sizeof(double) * 3 * N_atoms);
  types = (int*) calloc(sizeof(int), N_atoms);

  // Read the file
  ReadSCF(scf_file);
}



Atoms::~Atoms(){
  // Free memory
  free(coords);
  free(types);

}


int Atoms::GetNAtoms() { return N_atoms; }
int Atoms::GetNTypes() { return N_types; }

void Atoms::GetAtomsCoords(int index, double &x, double &y, double &z) {
  // Check if the index is valid
  if (index < 0 || index >= N_atoms) {
    cerr << "Error, atom index " << index << " out of " << N_atoms << endl;
    throw invalid_argument("Error, atom index out of range.\n");
  }

  x = coords[3 * index];
  y = coords[3 * index + 1];
  z = coords[3 * index + 2];
}


void Atoms::ReadSCF(string path_to_file, double alat) {

  // Read the input
  ifstream input(path_to_file.c_str());

  if (!input) {
    cerr << "Error while reading " << path_to_file << endl;
    throw invalid_argument("File not found!");
  }

  string line;
  bool reading_cell = false;
  int j;
  bool reading_atoms = true;
  double x, y, z;
  string atm_symb;
  vector<string> symbs;
  int current_type = 0;
  N_types = 0;
  while(getline(input, line)) {
    // Check if there is text in the line
    if (DEBUG_ATOMS) cout << "READING : " << line << endl;
    if (line.find("CELL") != string::npos) {

      if (DEBUG_ATOMS) cout << "READING THE CELL" << endl;
      
      // Read the cell
      for (int i = 0; i < 3; ++i) {
        if (! (input >> x >> y >> z)) {
          cerr << "Error, the cell vector must contain only three numbers" << endl;
          throw "";
        }
        unit_cell[3 * i] = x;
        unit_cell[3*i + 1] = y;
        unit_cell[3*i + 2] = z;

        if (DEBUG_ATOMS) 
          cout << "READING " << i << " => " << x << y << z << endl;
      }
    }
    if (line.find("ATOMIC") != string::npos) {
      if (DEBUG_ATOMS) cout << "READING THE ATOMS" << endl;

      for (int i = 0; i < N_atoms; ++i) {
        if (! (input >> atm_symb >> x >> y >> z)) {
          cerr << "Error, " << N_atoms << "atomic species plus three coordinates are reqired." << endl;
          throw "";
        }
        if (DEBUG_ATOMS) cout << "READING " << i << " => " << x << "\t" << y << '\t' << z;


        coords[3*i] = x;
        coords[3*i + 1] = y;
        coords[3*i + 2] = z;

        // Parse the atom symb
        for (j = 0; j < symbs.size(); ++j) {
          if (symbs.at(j) == atm_symb) break;
        }

        // Push a new symbol if not found
        if (j == symbs.size()) {
          // A new type found
          N_types++;
          symbs.push_back(atm_symb);
        }

        types[i] = j;
      }
    }
  }
  input.close();
}

void Atoms::GenerateSupercell(int Nx, int Ny, int Nz, Atoms * &new_target) {
  // Check if the supercell is odd
  if (Nx % 2 == 0 || Ny % 2 == 0 || Nz % 2 == 0) {
    cerr << "Error, an odd supercell neaded." << endl;
    throw "";
  }

  new_target = new Atoms(N_atoms * Nx * Ny * Nz);
  
  int id_x, id_y, id_z;
  for (int ix = 0; ix < Nx; ++ix) {
    id_x = (ix+1)  / 2;
    for (int k = 0; k< ix; ++k) id_x *= (-1);

    for(int iy = 0; iy < Ny; iy++) {
      id_y = (1 + iy)  / 2;
      for (int k = 0; k< iy; ++k) id_y *= (-1);

      for(int iz = 0; iz < Nz; iz++) {
        id_z = (1 + iz)  / 2;
        for (int k = 0; k< iz; ++k) id_z *= (-1);

        for (int j = 0; j < N_atoms; ++j) {
          new_target->types[j + N_atoms*iz + N_atoms*Nz*iy + N_atoms*Nz*Ny*ix] = types[j];
          for (int k = 0; k < 3; ++k)
            new_target->coords[k + 3*j + 3*N_atoms * iz + 3*N_atoms *Nz * iy + 3*N_atoms *Nz * Ny*ix] = 
              coords[k + 3*j] + id_x*unit_cell[k] + id_y*unit_cell[3 + k] + id_z*unit_cell[6 + k];
        }
      }
    }
  }
}



void Atoms::PrintCoords(void) {
  cout << endl;
  cout << "ATOMIC POSITIONS:" << endl;
  for (int i = 0; i < N_atoms; ++i) {
    cout << "TYPE: " << types[i] << "    COORDS: " << coords[3*i] << "   " << coords[3*i + 1] << 
      "   " << coords[3*i + 2] << endl;
  }
  cout << endl;
}
/* 
void Atoms::SetEnergyForce(double en, const double * forc) {
  // Allocate the memory for the forces
  if (forces == NULL) {
    forces = new double[GetNAtoms()];
  }

  energy = en;
  for (int i = 0; i < GetNAtoms() * 3; ++i) {
    forces[i] = forc[i];
  }
} */


int Atoms::GetNAtomsPerType(int type) {
  // Look how many atoms are in the given type
  int ret = 0;
  for (int i = 0; i < N_atoms; ++i) {
    if (type == types[i]) ret++;
  }
  return ret;
}