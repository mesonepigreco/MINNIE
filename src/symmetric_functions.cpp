#include "symmetric_functions.hpp"
#include <stdexcept>
#include <math.h>
#include <iostream>
#include <libconfig.h++>

using namespace std;
using namespace libconfig;



double cutoff_function(int cutoff_type, double Rc, double r) {
  // Check if the cutoff type is correct
  if (cutoff_type < 0 || cutoff_type > 1) {
    cerr << "ERROR: [CUTOFF FUNCTION] cutoff type is " << cutoff_type << endl;
    throw invalid_argument("cutoff_type not between [0,1]");
  }

  // Check if the r value is positive
  if (r < 0 || Rc < 0) {
    fprintf(stderr, "ERROR: [CUTOFF FUNCTION] r is %.8f and Rc is %.8f\n", r, Rc);
    throw invalid_argument("r must be positive");
  }

  double ret = 0;
  if (cutoff_type == 0 && r < Rc) {
    ret = 0.5 * (cos(M_PI * r / Rc) + 1);
  }
  else if(cutoff_type == 1 && r < Rc) {
    ret = tanh(1 - r/ Rc);
    ret = ret * ret * ret;
  }
  return ret;
}

double d_cutoff_function(int cutoff_type, double cutoff, double x) {
  if (x > cutoff) return 0;

  if (cutoff_type == 0) {
    return - 0.5 * M_PI / cutoff * sin(M_PI * x / cutoff);
  }
  
  double t, c;
  t = tanh(1 - x / cutoff);
  c = cosh(1 - x / cutoff);
  return -3 * t * t / (cutoff * c * c);
}

double G2_symmetry_func(double cutoff, double eta, double RS, int cutoff_type,
		      const double * coords, const int * atom_types, int N_atoms,
		      int atom_index, int type_1) {

  double ret = 0;

  double x0, y0, z0, x1, y1, z1, rij;

  // Get the position of the wanted atom
  x0 = coords[3*atom_index];
  y0 = coords[3*atom_index + 1];
  z0 = coords[3*atom_index + 2];
  
  for (int i = 0; i < N_atoms; ++i) {
    // Avoid to sum over the wrong atom types
    if (i == atom_index) continue;
    if (atom_types[i] != type_1) continue;

    // Get the position of the new atom
    x1 = coords[3*i];
    y1 = coords[3*i + 1];
    z1 = coords[3*i + 2];

    // Get the distance between the two atoms
    rij = (x0 - x1) * (x0 - x1) +
      (y0 - y1) * (y0 - y1) +
      (z0 - z1) * (z0 - z1);

    if (rij > cutoff * cutoff) continue;

    //cout << "ATOM: " << atom_index << " "<< i << " rij: " << rij << endl; 
    rij = sqrt(rij);

    //cerr << endl;
    //cerr << "RIJ:" << rij << " RS: " << RS << " ETA: " << eta << " CUTOFF: " << cutoff_function(cutoff_type, cutoff, rij) << endl; 
    //cerr << "COORDS1: " << x0 << " " << y0 << " " << z0 << "  |  " << x1 << " " << y1 << " " << z1 << endl;

    ret += exp(- eta * (rij - RS) * (rij - RS)) * cutoff_function(cutoff_type, cutoff, rij);
  }

  return ret;
}


double DG2_DX(double cutoff, double eta, double RS, int cutoff_type, 
  const double * coords, const int * atom_types, int N_atoms,
  int atom_index, int type_1, int d_atm_index, int d_coord_index) {
    
    // Atom index
    double ret = 0;
    double ri0 = 0;
    double dr_dx = 0;
    if (atom_index != d_atm_index) {
      // Check if the dest type is of the same type of the given function
      if (atom_types[d_atm_index] != type_1) return 0;

      ri0 = (coords[3*d_atm_index] - coords[3*atom_index]) * (coords[3*d_atm_index] - coords[3*atom_index]) +
        (coords[3*d_atm_index+1] - coords[3*atom_index+1]) * (coords[3*d_atm_index+1] - coords[3*atom_index+1]) +
        (coords[3*d_atm_index+2] - coords[3*atom_index+2]) * (coords[3*d_atm_index+2] - coords[3*atom_index+2]);

      ri0 = sqrt(ri0);
      dr_dx = (coords[3*d_atm_index+d_coord_index] - coords[3*atom_index+d_coord_index]) / ri0;
      ret = -2*eta *(ri0 - RS) * exp( -eta * (ri0 - RS) * (ri0 - RS)) * cutoff_function(cutoff_type, cutoff, ri0);
      ret += d_cutoff_function(cutoff_type, cutoff, ri0) * exp(-eta * (ri0 - RS) * (ri0 - RS));
      ret *= dr_dx;


      //cerr << "RIJ: " << ri0 << " DCUT: " << d_cutoff_function(cutoff_type, cutoff, ri0) << endl;

      return ret;
    } 

    double super_ret = 0;
    
    for (int i = 0; i < N_atoms; ++i) {
      if (i == atom_index) continue;
      if (atom_types[i] != type_1) continue;
      ri0 = (coords[3*d_atm_index] - coords[3*i]) * (coords[3*d_atm_index] - coords[3*i]) +
        (coords[3*d_atm_index+1] - coords[3*i+1]) * (coords[3*d_atm_index+1] - coords[3*i+1]) +
        (coords[3*d_atm_index+2] - coords[3*i+2]) * (coords[3*d_atm_index+2] - coords[3*i+2]);

      if (ri0 > cutoff * cutoff) continue;

      ri0 = sqrt(ri0);
      dr_dx = (coords[3*d_atm_index+d_coord_index] - coords[3*i+d_coord_index]) / ri0;
      ret = -2*eta *(ri0 - RS)* exp( -eta * (ri0 - RS) * (ri0 - RS)) * cutoff_function(cutoff_type, cutoff, ri0);
      ret += d_cutoff_function(cutoff_type, cutoff, ri0) * exp(-eta * (ri0 - RS) * (ri0 - RS));
      ret *= dr_dx; 
      super_ret += ret;
    }
    return super_ret;
}


double G4_symmetry_func(double cutoff, double zeta, double eta, int lambda, int cutoff_type,
		      const double * coords, const int * atom_types, int N_atoms,
		      int atom_index, int type_1, int type_2) {

  double ret = 0;

  double x0, y0, z0, x1, y1, z1;
  double x2, y2, z2;
  double rij, rjk, cos_theta;

  // Get the position of the wanted atom
  x0 = coords[3*atom_index];
  y0 = coords[3*atom_index + 1];
  z0 = coords[3*atom_index + 2];
  
  for (int i = 0; i < N_atoms; ++i) {
    // Avoid to sum over the wrong atom types
    if (i == atom_index) continue;
    if (atom_types[i] != type_1) continue;

    // Get the position of the new atom
    x1 = coords[3*i];
    y1 = coords[3*i + 1];
    z1 = coords[3*i + 2];

    // Get the distance between the two atoms
    rij = (x0 - x1) * (x0 - x1) +
      (y0 - y1) * (y0 - y1) +
      (z0 - z1) * (z0 - z1);

    if (rij > cutoff * cutoff) continue;
    rij = sqrt(rij);

    // Avoid the sum over the other atom if we are out of the cutoff region
    int start_pos = (type_1 == type_2)? i : 0;
    for (int k = start_pos; k < N_atoms; ++k) {
      if (k == i) continue;
      if (k == atom_index) continue;
      if (atom_types[k] != type_2) continue;

      // Get the position of the new atom
      x2 = coords[3*k];
      y2 = coords[3*k + 1];
      z2 = coords[3*k + 2];

      // Get the distance between the two atoms
      rjk = (x0 - x2) * (x0 - x2) +
	(y0 - y2) * (y0 - y2) +
	(z0 - z2) * (z0 - z2);
      rjk = sqrt(rjk);

      if (rjk > cutoff) continue;

      // Get the angle
      cos_theta = (x1-x0) * (x2-x0) +
	(y1 - y0) * (y2 - y0) +
	(z1 - z0) * (z2 - z0);
      cos_theta /= rij * rjk;

      ret += pow(1 + lambda * cos_theta, zeta) *
	exp(-eta * (rij*rij + rjk*rjk)) *
	cutoff_function(cutoff_type, cutoff, rij) * cutoff_function(cutoff_type, cutoff, rjk);
    }
  }

  // Apply the normalization
  ret *= pow(2, 1 - zeta);
  return ret;
}



double DG4_DX(double cutoff, double zeta, double eta, int lambda, int cutoff_type, const double * coords, const int * atom_types, 
  int N_atoms, int atom_index, int type_1, int type_2, int d_atm_index, int d_coord_index) {  
    
    double res = 0;
    double x0, y0, z0, x1, y1, z1;
    double x2, y2, z2;
    double rij, rjk, cos_theta;
    double d_rij_dx,  d_rjk_dx,  d_rij2_dx,  d_rjk2_dx;
    double d_costheta_dx;

    // Get the position of the wanted atom
    x0 = coords[3*atom_index];
    y0 = coords[3*atom_index + 1];
    z0 = coords[3*atom_index + 2];
      
    // Check if the atom of the derivative is the self atom.
    if (d_atm_index == atom_index) {
      // Compute similar as the G4 

      for (int i = 0; i < N_atoms; ++i) {
        // Avoid to sum over the wrong atom types
        if (i == atom_index) continue;
        if (atom_types[i] != type_1) continue;

        // Get the position of the new atom
        x1 = coords[3*i];
        y1 = coords[3*i + 1];
        z1 = coords[3*i + 2];

        // Get the distance between the two atoms
        rij = (x0 - x1) * (x0 - x1) +
          (y0 - y1) * (y0 - y1) +
          (z0 - z1) * (z0 - z1);

        if (rij > cutoff * cutoff) continue;

        rij = sqrt(rij);
        d_rij2_dx = 2 * ( coords[3*d_atm_index + d_coord_index] - coords[3*i + d_coord_index]);
        d_rij_dx = d_rij2_dx / (2 * rij);

        // Avoid the sum over the other atom if we are out of the cutoff region
        int start_pos = (type_1 == type_2)? i : 0;
        for (int k = start_pos; k < N_atoms; ++k) {
          if (k == i) continue;
          if (k == atom_index) continue;
          if (atom_types[k] != type_2) continue;

          // Get the position of the new atom
          x2 = coords[3*k];
          y2 = coords[3*k + 1];
          z2 = coords[3*k + 2];

          // Get the distance between the two atoms
          rjk = (x0 - x2)*(x0 - x2) + (y0 - y2)*(y0 - y2) + (z0 - z2)*(z0 - z2);
          if (rjk > cutoff*cutoff) continue;
          rjk = sqrt(rjk);
          d_rjk2_dx = 2 * ( coords[3*d_atm_index + d_coord_index] - coords[3*k + d_coord_index]);
          d_rjk_dx = d_rjk2_dx / (2 * rjk);


          // Get the angle
          cos_theta = (x1-x0)*(x2-x0) + (y1 - y0)*(y2 - y0) + (z1 - z0)*(z2 - z0);
          cos_theta /= rij * rjk;
          d_costheta_dx = (2*coords[3*d_atm_index + d_coord_index] - coords[3*i + d_coord_index] - coords[3*k + d_coord_index]);
          d_costheta_dx -= cos_theta * (rij * d_rjk_dx + rjk * d_rij_dx);
          d_costheta_dx /= rij*rjk;

          // Now compute the derivative
          // The first part is the derivative with respect to 1 + lambda*cos(theta)
          res += pow(1 + lambda * cos_theta, zeta - 1) * lambda*zeta * d_costheta_dx * 	
            exp(-eta * (rij*rij + rjk*rjk)) * 
            cutoff_function(cutoff_type, cutoff, rij) * cutoff_function(cutoff_type, cutoff, rjk);

          // Now we can derivate with respect to the exponential
          res -= pow(1 + lambda * cos_theta, zeta) * 
            exp(-eta * (rij*rij + rjk*rjk)) * eta * (d_rij2_dx + d_rjk2_dx) *
            cutoff_function(cutoff_type, cutoff, rij) * cutoff_function(cutoff_type, cutoff, rjk);
          
          // Now we can derivate with respect to the cutoff
          res += pow(1 + lambda * cos_theta, zeta) * 
            exp(-eta * (rij*rij + rjk*rjk)) *
            (d_cutoff_function(cutoff_type, cutoff, rij) * d_rij_dx * cutoff_function(cutoff_type, cutoff, rjk) + 
            cutoff_function(cutoff_type, cutoff, rij)  * d_cutoff_function(cutoff_type, cutoff, rjk)* d_rjk_dx);
        }
      }
      res *= pow(2, 1 - zeta);
      return res;
    }

    // Here the selected atom is not the central one.

    // Check if the atom is of the proper type
    int other_typ = type_1;
    if (atom_types[d_atm_index] != type_1) {
      other_typ = type_2;
      if (atom_types[d_atm_index != type_2]) return 0;
    }
    // Get the coordinates for the atom of the derivative
    x1 = coords[3*d_atm_index];
    y1 = coords[3*d_atm_index + 1];
    z1 = coords[3*d_atm_index + 2];

    rij = (x0 - x1) * (x0 - x1) +
      (y0 - y1) * (y0 - y1) +
      (z0 - z1) * (z0 - z1);
    rij = sqrt(rij);
    d_rij2_dx = 2 * ( coords[3*d_atm_index + d_coord_index] - coords[3*atom_index + d_coord_index]);
    d_rij_dx = d_rij2_dx / (2 * rij);

    // If my atom is more distant than the cutoff, return 0
    if (rij > cutoff) return 0;

    // Now sum over all the remaining atoms
    for (int k = 0; k < N_atoms; ++k) {
      if (k == d_atm_index) continue;
      if (k == atom_index) continue;
      if (atom_types[k] != other_typ) continue;

      // Get the position of the new atom
      x2 = coords[3*k];
      y2 = coords[3*k + 1];
      z2 = coords[3*k + 2];

      // Get the distance between the two atoms
      rjk = (x0 - x2)*(x0 - x2) + (y0 - y2)*(y0 - y2) + (z0 - z2)*(z0 - z2);
      rjk = sqrt(rjk);

      if (rjk > cutoff) continue;

      // Get the angle
      cos_theta = (x1-x0)*(x2-x0) + (y1 - y0)*(y2 - y0) + (z1 - z0)*(z2 - z0);
      cos_theta /= rij * rjk;
      d_costheta_dx = (coords[3*k + d_coord_index] - coords[3*atom_index + d_coord_index]);
      d_costheta_dx -= cos_theta * (rjk * d_rij_dx);
      d_costheta_dx /= rij*rjk;

      // Now compute the derivative
      // The first part is the derivative with respect to 1 + lambda*cos(theta)
      res += pow(1 + lambda * cos_theta, zeta - 1) * lambda*zeta * d_costheta_dx * 	
        exp(-eta * (rij*rij + rjk*rjk)) * 
        cutoff_function(cutoff_type, cutoff, rij) * cutoff_function(cutoff_type, cutoff, rjk);

      // Now we can derivate with respect to the exponential
      res -= pow(1 + lambda * cos_theta, zeta) * 
        exp(-eta * (rij*rij + rjk*rjk)) * eta * (d_rij2_dx) *
        cutoff_function(cutoff_type, cutoff, rij) * cutoff_function(cutoff_type, cutoff, rjk);
      
      // Now we can derivate with respect to the cutoff
      res += pow(1 + lambda * cos_theta, zeta) * 
        exp(-eta * (rij*rij + rjk*rjk)) *
        d_cutoff_function(cutoff_type, cutoff, rij) * d_rij_dx * cutoff_function(cutoff_type, cutoff, rjk);
    }
    res *= pow(2, 1 - zeta);
    return res; 
}

SymmetricFunctions::SymmetricFunctions() {
  // Prepare the symmetric functions
  N_G2 = 0;
  N_G4 = 0;
  cutoff_radius = 6; // 6 Angstrom
  cutoff_function_type = 0; 
};

int SymmetricFunctions::get_n_g2(void) {
  return G2_RS.size();
}
int SymmetricFunctions::get_n_g4(void) {
  return G4_ETA.size();
}

void SymmetricFunctions::SetupCutoffFunction(int type, double cutoff_value) {
  // Check if the type is or 0 or 1
  if (type != 0 && type != 1) 
    throw invalid_argument("ERROR, the cutoff type can be only or 0 or 1.\n");

  if (cutoff_value <= 0) 
    throw invalid_argument("ERROR, the cutoff radius cannot be negative.\n");
    
  cutoff_radius = cutoff_value;
  cutoff_function_type = type;
};

double SymmetricFunctions::get_cutoff() {
  return cutoff_radius;
}

int SymmetricFunctions::get_cutoff_type() { 
  return cutoff_function_type;
}



int SymmetricFunctions::GetTotalNSym(int N_types) {
  return G2_RS.size() * N_types + G4_ZETA.size() * (N_types * (N_types + 1)) / 2;
}
int SymmetricFunctions::GetG2Sym(int N_types) {
  return G2_RS.size() * N_types;
}
int SymmetricFunctions::GetG4Sym(int N_types) {
  return G4_ZETA.size() * (N_types * (N_types + 1)) / 2.;
}

void SymmetricFunctions::GetSymmetricFunctionsInput(const double * coords, const int * atm_types,int N_atoms, int N_types,
						   int atom_index, double * sym_values) {


  // Use the G2
  for (int i = 0; i < N_G2; ++i) {
    // Cycle over the types
    for (int j = 0; j < N_types; ++j) {
      sym_values[N_types * i + j] = G2_symmetry_func(cutoff_radius, G2_ETA.at(i),
						     G2_RS.at(i), cutoff_function_type,
						     coords, atm_types, N_atoms, atom_index, j);
    }
  }

  // Use the G4
  int index = 0;
  for (int i = 0; i < N_G4; ++i) {
    for (int j = 0; j < N_types; ++j) {
      int counter_j = 0;
      for (int k = j; k < N_types; ++k) {
	index = N_types*(N_types+1)/2 * i + counter_j + k;
	sym_values[N_types*N_G2 + index] =
	  G4_symmetry_func(cutoff_radius, G4_ZETA.at(i), G4_ETA.at(i),
			   G4_LAMBDA.at(i), cutoff_function_type,
			   coords, atm_types, N_atoms, atom_index, j, k);
      }
      counter_j += N_types - j;
    }
  }
}


void SymmetricFunctions::GetSymmetricFunctions(Atoms * structure, int Nx, int Ny, int Nz, double * sym_values) {
  Atoms * supercell;

  // Prepare the supercell
  structure->GenerateSupercell(Nx, Ny, Nz, supercell);

  int nat_sc = supercell->GetNAtoms();
  int nat = structure->GetNAtoms();
/* 
  cout << "UNIT CELL:" << endl;
  for (int i = 0; i < 3; ++i) {
    cout << structure->unit_cell[3*i] << "  " << structure->unit_cell[3*i + 1] << "  " <<  structure->unit_cell[3*i + 2] << endl;
  }

  // Print the supercell structure
  cout << endl << "COORDS:" << endl;
  for (int i = 0; i < nat_sc; ++i) {
    cout << i << "   " << supercell->coords[3*i] << "  " << supercell->coords[3*i+1] << "   " << supercell->coords[3*i + 2] << endl;
  } */

  // For each atom in the current structure compute the symmetric functions
  int n_sym = GetTotalNSym(structure->GetNTypes());

  for (int i = 0; i < nat; ++i) {
    GetSymmetricFunctionsInput(supercell->coords, supercell->types, nat_sc, structure->GetNTypes(), i, sym_values + i * n_sym);
  }
  delete supercell;
}

void SymmetricFunctions::GetDerivatives(Atoms * structure, int Nx, int Ny, int Nz, int d_atm_index, int d_cart_coord, double * sym_diff) {
  Atoms * supercell;

  // Prepare the supercell
  structure->GenerateSupercell(Nx, Ny, Nz, supercell);

  int nat_sc = supercell->GetNAtoms();
  int nat = structure->GetNAtoms();
  int N_types = structure->GetNTypes();

  double * coords = supercell->coords;
  double * uc_coords = structure->coords;
  int * atm_types = supercell->types;

  // For each atom in the current structure compute the symmetric functions
  int n_sym = GetTotalNSym(structure->GetNTypes());

  int n_replicas = Nx * Ny * Nz;

  // Compute the derivatives of the C2
  int excluded = 0;
  for (int atom_index = 0; atom_index < nat; ++atom_index){ 
    // G2 symmetry functions initialization
    for (int i = 0; i < N_G2; ++i) {
      for (int j = 0; j < N_types; ++j) {
          sym_diff[n_sym *atom_index+ N_types * i + j] = 0;
      }
    }

    // G4 initialization
    int index = 0;
    for (int i = 0; i < N_G4; ++i) {
      for (int j = 0; j < N_types; ++j) {
        int counter_j = 0;
        for (int k = j; k < N_types; ++k) {
          index = N_types*N_types * i + counter_j + k + n_sym *atom_index;
          sym_diff[N_types*N_G2 + index] = 0;
        }
        counter_j += N_types - j;
      }
    }

    for (int k = 0; k < n_replicas; ++ k) {
      int d_atm_index_replica = d_atm_index + k * nat;

      // Check if the function goes outside the cutoff with the atom derived.
      double r2 = 0;
      for (int i = 0; i < 3; ++i) r2 += (uc_coords[3 * atom_index + i] - coords[3*d_atm_index_replica + i]) * (uc_coords[3 * atom_index + i] - coords[3*d_atm_index_replica + i]);
      if (r2 > cutoff_radius * cutoff_radius) {
        excluded += 1;
        continue;
      } 

      // G2 symmetry functions
      for (int i = 0; i < N_G2; ++i) {
        // Cycle over the types
        for (int j = 0; j < N_types; ++j) {
            sym_diff[n_sym *atom_index+ N_types * i + j] += DG2_DX(cutoff_radius, G2_ETA.at(i),
                      G2_RS.at(i), cutoff_function_type,
                      coords, atm_types, nat_sc, atom_index, j, d_atm_index_replica, d_cart_coord);
        }
      }
      
      

      // Use the G4
      int index = 0;
      for (int i = 0; i < N_G4; ++i) {
        for (int j = 0; j < N_types; ++j) {
          int counter_j = 0;
          for (int k = j; k < N_types; ++k) {
      index = N_types*N_types * i + counter_j + k + n_sym *atom_index;
      sym_diff[N_types*N_G2 + index] +=
        DG4_DX(cutoff_radius, G4_ZETA.at(i), G4_ETA.at(i),
            G4_LAMBDA.at(i), cutoff_function_type,
            coords, atm_types, nat_sc, atom_index, j, k, d_atm_index_replica, d_cart_coord);
          }
          counter_j += N_types - j;
        }
      }
    }
  }

  //cout << "Excluded: " << excluded * 100./(float) nat << " \%  cutoff: " << cutoff_radius << " A" << endl;

  delete supercell;
}



void SymmetricFunctions::PrintInfo(void) {
  cout << "SYMMETRIC FUNCTIONS" << endl;
  cout << "===================" << endl;
  cout << endl;
  cout << "Cutoff type : " << cutoff_function_type << endl;
  cout << "Cutoff radius : " << cutoff_radius << " Angstrom" << endl;
  cout << endl;
  cout << "Number of G2 functions : " << N_G2 << endl;
  cout << "Values of eta (G2):" << endl;
  for (int i = 0; i<N_G2; ++i) cout << G2_ETA.at(i) << "\t";
  cout << endl << "Values of Rs (G2):" << endl;
  for (int i = 0; i<N_G2; ++i) cout << G2_RS.at(i) << "\t";
  cout << endl << endl;

  cout << "Number of G4 functions : " << N_G4 << endl;
  cout << "Values of eta (G4):" << endl;
  for (int i = 0; i<N_G4; ++i) cout << G4_ETA.at(i) << "\t";
  cout << endl << "Values of zeta (G4):" << endl;
  for (int i = 0; i<N_G4; ++i) cout << G4_ZETA.at(i) << "\t";
  cout << endl << "Values of lambda (G4):" << endl;
  for (int i = 0; i<N_G4; ++i) cout << G4_LAMBDA.at(i) << "\t";
  cout << endl;
  cout << endl;
}


void SymmetricFunctions::AddG2Function(double rs, double eta) {
  N_G2 += 1;
  G2_RS.push_back(rs);
  G2_ETA.push_back(eta);

  // Check if there is an inconsistency
  if (N_G2 != G2_RS.size() || N_G2 != G2_ETA.size()) {
    cerr << "Error while appending a G2 function" << endl;
    cerr << "The N_G2 does not match the size of the parameters" << endl;
    throw "";
  }
}

void SymmetricFunctions::PopG2Function(int index) {
  // Check if the index is good
  if (index >= N_G2) {
    cerr << "Error, cannot remove the G2 function " << index << endl;
    cerr << "Out of range (size " << N_G2 << ")" << endl;
    throw "";
  } else if (index < 0) {
    cerr << "Error, invalid pop index for symmetric function " << index << endl;
    cerr << "Inside: PopG2Function" << endl;
    throw "";
  }
  
  for (int i = index; i < G2_ETA.size()-1; ++i) {
    G2_ETA.at(i) = G2_ETA.at(i+1);
    G2_RS.at(i) = G2_RS.at(i+1);
  }
  G2_ETA.pop_back();
  G2_RS.pop_back();
  N_G2 -= 1;
}

void SymmetricFunctions::AddG4Function(double eta, double zeta, int lambda) {
  N_G4 += 1;
  G4_LAMBDA.push_back(lambda);
  G4_ZETA.push_back(zeta);
  G4_ETA.push_back(eta);

  if (lambda != 1 && lambda != -1) {
    cerr << "WARNING: G4 lambda sould be +- 1" << endl;
    cerr << "         yor lambda = " << lambda << endl;
    cerr << "         please check if this is really what you want." << endl;
  }

  // Check if there is an inconsistency
  if (N_G4 != G4_ZETA.size() || N_G4 != G4_ETA.size() || N_G4 != G4_LAMBDA.size()) {
    cerr << "Error while appending a G4 function" << endl;
    cerr << "The N_G4 does not match the size of the parameters" << endl;
    cerr << "Counted g4: " << N_G4 << " | real g4 (zetas): " << G4_ZETA.size() << endl;
    cerr << "Counted g4: " << N_G4 << " | real g4 (etas): " << G4_ETA.size() << endl;
    cerr << "Counted g4: " << N_G4 << " | real g4 (lambdas): " << G4_LAMBDA.size() << endl;
    throw "";
  }
}


void SymmetricFunctions::PopG4Function(int index) {
  // Check if the index is good
  if (index >= N_G4) {
    cerr << "Error, cannot remove the G4 function " << index << endl;
    cerr << "Out of range (size " << N_G4 << ")" << endl;
    throw "";
  } else if (index < 0) {
    cerr << "Error, invalid pop index for symmetric function " << index << endl;
    cerr << "Inside: PopG4Function" << endl;
    throw "";
  }
  
  for (int i = index; i < G4_ETA.size()-1; ++i) {
    G4_ETA.at(i) = G4_ETA.at(i+1);
    G4_ZETA.at(i) = G4_ZETA.at(i+1);
    G4_LAMBDA.at(i) = G4_LAMBDA.at(i+1);
  }
  G4_ETA.pop_back();
  G4_ZETA.pop_back();
  G4_LAMBDA.pop_back();
  N_G4 -= 1;
}


void SymmetricFunctions::LoadFromCFG(const char * config_file) {
  Config cfg;

  try {
    cfg.readFile(config_file);
  } catch (const FileIOException &e) {
    cerr << "Error while reading the file " << config_file << endl;
    throw;
  } catch(const ParseException &e) {
    cerr << "Error while parsing the file: " << config_file << endl;
    cerr << "Line:  " << e.getLine() << endl;
    cerr << e.getError() << endl;
    throw;
  }

  if (! cfg.exists(SYMMFUNC_ENVIRON)) {
    cerr << "Error, the symmetric functions environment has not been setted." << endl;
    cerr << "KEYWORD: " << SYMMFUNC_ENVIRON << endl;
    throw;
  }

  const Setting& root = cfg.getRoot();
  const Setting& symm_func_set = root[SYMMFUNC_ENVIRON];

  // Read the G2 and G4
  try {
    if (!symm_func_set.lookupValue(SYMMFUNC_NG2, N_G2)) {
      cerr << "Error, you must specify how many G2 function to use" << endl;
      throw;
    }
  } catch (...) {
    cerr << "Error while setting the N_G2, please check carefully the type." << endl;
  }
  try {
    if (!symm_func_set.lookupValue(SYMMFUNC_NG4, N_G4)) {
      cerr << "Error, you must specify how many G4 function to use" << endl;
      throw;
    }
  } catch (...) {
    cerr << "Error while setting the N_G4, please check carefully the type." << endl;
  }

  // Read the G2 functions
  if (N_G2 > 0) {
    try {
      const Setting & s_g2eta = symm_func_set[SYMMFUNC_G2ETA];
      const Setting & s_g2rs = symm_func_set[SYMMFUNC_G2RS];

      // Check the length
      if (s_g2eta.getLength() != N_G2) {
        cerr << "Error, the number of " << SYMMFUNC_G2ETA << " values must match " << N_G2 << endl;
        throw "";
      }
      if (s_g2rs.getLength() != N_G2) {
        cerr << "Error, the number of " << SYMMFUNC_G2RS << " values must match " << N_G2 << endl;
        throw "";
      }

      // Add the G2 functions to the network
      for (int i = 0; i < N_G2; ++i) {
        G2_ETA.push_back(s_g2eta[i]);
        G2_RS.push_back(s_g2rs[i]);
      }
    } catch (const SettingException &e) {
      cerr << "Error, check setting " << e.getPath() << " carefully" << endl;
      throw;
    }
  }

  // Add the G4 functions
  if (N_G4 > 0) {
    try {
      const Setting & s_g4eta = symm_func_set[SYMMFUNC_G4ETA];
      const Setting & s_g4zeta = symm_func_set[SYMMFUNC_G4ZETA];
      const Setting & s_g4lambda = symm_func_set[SYMMFUNC_G4LAMBDA];


      // Check the length
      if (s_g4eta.getLength() != N_G4) {
        cerr << "Error, the number of " << SYMMFUNC_G4ETA << " values must match " << N_G2 << endl;
        throw "";
      }
      if (s_g4zeta.getLength() != N_G4) {
        cerr << "Error, the number of " << SYMMFUNC_G4ZETA<< " values must match " << N_G2 << endl;
        throw "";
      }
      if (s_g4lambda.getLength() != N_G4) {
        cerr << "Error, the number of " << SYMMFUNC_G4LAMBDA<< " values must match " << N_G2 << endl;
        throw "";
      }


      // Add the G2 functions to the network
      for (int i = 0; i < N_G4; ++i) {
        G4_ETA.push_back(s_g4eta[i]);
        G4_ZETA.push_back(s_g4zeta[i]);
        G4_LAMBDA.push_back(s_g4lambda[i]);
      }
    } catch (const SettingException &e) {
      cerr << "Error, check setting " << e.getPath() << " carefully" << endl;
      throw;
    }
  }

  // Setup the cutoff function
  cutoff_function_type = 0;
  if (!symm_func_set.exists(SYMMFUNC_CUTOFF)) {
    cerr << "Error, the " << SYMMFUNC_CUTOFF << " is required" << endl;
    throw;
  }
  try {
    cutoff_radius = (double) symm_func_set.lookup(SYMMFUNC_CUTOFF);
    if (symm_func_set.exists(SYMMFUNC_CUTOFFTYPE))
      symm_func_set.lookupValue(SYMMFUNC_CUTOFFTYPE, cutoff_function_type);
  } catch (const SettingException & e) {
    cerr << "Error, check setting " << e.getPath() << " for proper typing." << endl;
    throw;
  }
}

void SymmetricFunctions::SaveToCFG(const char * cfg_filename) {
  // Get the root of the config
  Config cfg;
  Setting& root = cfg.getRoot();

  // Add the main environment
  Setting& main_env = root.add(SYMMFUNC_ENVIRON, Setting::TypeGroup);

  cout << "Saving the number and the cutoff" << endl;
  // Add the number of symmetric functions
  main_env.add(SYMMFUNC_NG2, Setting::TypeInt) = N_G2;
  main_env.add(SYMMFUNC_NG4, Setting::TypeInt) = N_G4;
  main_env.add(SYMMFUNC_CUTOFF, Setting::TypeFloat) = cutoff_radius;
  main_env.add(SYMMFUNC_CUTOFFTYPE, Setting::TypeInt) = cutoff_function_type;


  // Add the G2 symmetric functions
  Setting& g2_rs = main_env.add(SYMMFUNC_G2RS, Setting::TypeArray);
  Setting& g2_eta = main_env.add(SYMMFUNC_G2ETA, Setting::TypeArray);

  for (int i = 0; i < N_G2; ++i) {
    g2_rs.add(Setting::TypeFloat) = G2_RS.at(i);
    g2_eta.add(Setting::TypeFloat) = G2_ETA.at(i);
  }

  // Add the G4 functions
  Setting& g4_zeta = main_env.add(SYMMFUNC_G4ZETA, Setting::TypeArray);
  Setting& g4_eta = main_env.add(SYMMFUNC_G4ETA, Setting::TypeArray);
  Setting& g4_lambda = main_env.add(SYMMFUNC_G4LAMBDA, Setting::TypeArray);

  for (int i = 0; i < N_G4; ++i) {
    g4_zeta.add(Setting::TypeFloat) = G4_ZETA.at(i);
    g4_eta.add(Setting::TypeFloat) = G4_ETA.at(i);
    g4_lambda.add(Setting::TypeInt) = G4_LAMBDA.at(i);
  }

  // write to file
  cfg.writeFile(cfg_filename);
}


SymmetricFunctions::~SymmetricFunctions(){
  // Nothing to do.
}


void SymmetricFunctions::GetG2Parameters(int index, double &rs, double &eta){
  if (index > N_G2) {
    cerr << "Error, the maximum number of G2 functions is " << N_G2 << endl;
    cerr << "       requested index " << index << endl;
    throw "";
  }
  if (index <0) index = N_G2 - index;

  rs = G2_RS.at(index);
  eta = G2_ETA.at(index);
}

void SymmetricFunctions::GetG4Parameters(int index, double &zeta, double &eta, int& lambda){
  if (index > N_G4) {
    cerr << "Error, the maximum number of G2 functions is " << N_G4 << endl;
    cerr << "       requested index " << index << endl;
    throw "";
  }
  if (index <0) index = N_G2 - index;

  zeta = G4_ZETA.at(index);
  eta = G4_ETA.at(index);
  lambda = G4_LAMBDA.at(index);
}

void SymmetricFunctions::EditG2Function(int index, double new_rs, double new_eta){
  if (index > N_G2) {
    cerr << "Error, the maximum number of G2 functions is " << N_G2 << endl;
    cerr << "       requested index " << index << endl;
    throw "";
  }
  if (index <0) index = N_G2 - index;

  G2_RS.at(index) = new_rs;
  G2_ETA.at(index) = new_eta;
}


void SymmetricFunctions::EditG4Function(int index, double zeta, double eta, int lambda){
  if (index > N_G4) {
    cerr << "Error, the maximum number of G2 functions is " << N_G4 << endl;
    cerr << "       requested index " << index << endl;
    throw "";
  }
  if (index <0) index = N_G2 - index;

  G4_ZETA.at(index) = zeta;
  G4_ETA.at(index) = eta;
  G4_LAMBDA.at(index) = lambda;
}