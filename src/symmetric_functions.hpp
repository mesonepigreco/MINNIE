#ifndef SYM_FUNC
#define SYM_FUNC
#include <vector>
#include <string>
#include <iostream>
#include "atoms.hpp"
#include <libconfig.h++>

using namespace std;
using namespace libconfig;

// Define the CFG options
#define SYMMFUNC_ENVIRON "SymmetricFunctions"
#define SYMMFUNC_NG2 "n_g2"
#define SYMMFUNC_NG4 "n_g4"
#define SYMMFUNC_G2ETA "g2_eta"
#define SYMMFUNC_G2RS "g2_Rs"
#define SYMMFUNC_G4ETA "g4_eta"
#define SYMMFUNC_G4ZETA "g4_zeta"
#define SYMMFUNC_G4LAMBDA "g4_lambda"
#define SYMMFUNC_CUTOFF "cutoff_radius"
#define SYMMFUNC_CUTOFFTYPE "cutoff_type"
#define SUPERCELL_NX "N_sup_x"
#define SUPERCELL_NY "N_sup_y"
#define SUPERCELL_NZ "N_sup_z"



class SymmetricFunctions {
private :
  // The number of symmetric functions
  int N_G2, N_G4;

  // The cutoff radius
  double cutoff_radius;

  // Type of the cutoff function (may be 0 or 1)
  int cutoff_function_type;

  // The info about the symmetry functions
  // They are vector as the number of symmetry function may change at the begining during
  // the optimization
  vector<double> G2_RS, G2_ETA;
  vector<double> G4_ZETA, G4_ETA;
  vector<int> G4_LAMBDA;


public:
  SymmetricFunctions();
  ~SymmetricFunctions();

  // Get the info on the specific symmetric function (G2)
  void GetG2Parameters(int index, double &rs, double &eta);

  // Get the info on the specific symmetric function (G4)
  void GetG4Parameters(int index, double &zeta, double &eta, int &lambda);

  // Edit the G2 symmetric function given the index
  void EditG2Function(int index, double new_rs, double new_eta);

  // Edit the G4 symmetric function given the index
  void EditG4Function(int index, double new_zeta, double new_eta, int new_lambda);

  /*
   * Get the value of the symmetric functions, given the cartesian coordinates of the systems.
   *
   * Parameters
   * ----------
   *
   *  - coords : 3 x N_atoms array of the cartesian coordinates
   *  - atm_types : N_atoms array with the index of the type of the atoms
   *  - N_atoms : Number of the atoms in the cartesian input.
   *  - N_types : Number of different atomic types in the structure
   *  - atom_index : the index of the atom for which you want to carry out the symmetric functions
   *  - sym_values : output array (must already been initialized) of the values of the symmetry functions for the given atom_index
   *
   * The length of the output array 'sym_values' lenght must be equal to the output of
   * the function GetTotalNSym()
   */
  void  GetSymmetricFunctionsInput(const double * coords, const int * atm_types, int N_atoms, int N_types,
				  int atom_index, double * sym_values);


  /*
   * This method is used to evaluate the symmetric functions in a given atomic object
   * It builds a supercell near the structure of the given size to correctly identify nearby atoms,
   * And returns the list of the symmetric function values for each atoms.
   * The symmetric functiona array must be already initialized, of size N_atoms * N_syms
   * 
   * Parameters
   * ----------
   * 
   *  structure : Pointer to atom structure
   *      The structure on which you want to compute the symmetric values
   *  Nx, Ny, Nz : integers
   *      The supercell size (must be odd) that surrounds the given structure.
   *  sym_values : pointer to double precisions
   *      The symmetric functions per each atom in the original structure.
   * 
   */
  void GetSymmetricFunctions(Atoms * structure, int Nx, int Ny, int Nz, double * sym_values, int N_types = -1);

  /*
   * This method, in a way similar to the previous one, computes the derivatives of the symmetric functions
   * with respect to the given specific atomic position.
   * 
   * Parameters
   * ----------
   * 
   * structure : Pointer to atom structure
   *      The structure on which the symmetric functions are evaluated
   * Nx, Ny, Nz : integers
   *      The supercell size (must be odd) that surronds the structure.
   * atm_index : integer
   *      The index of the atom that is mooving.
   * cart_coord : integer
   *      The integer of the cartesian coordinate that is mooved.
   * sym_diff : pointer to doulble precisions
   *      The derivatives of each symmetric function with respect to the specified atomic displacement.
   * 
   */
  void GetDerivatives(Atoms * structure, int Nx, int Ny, int Nz, int atm_index, int cart_coord, double * sym_diff, int N_types = -1);
  
  
  // Add a G2 function
  void AddG2Function(double rs, double eta);
  void PopG2Function(int index);

  // Add a G4 function
  void AddG4Function(double eta, double zeta, int lambda);
  void PopG4Function(int index);

  
  /* SETUP THE CUTOFF FUNCTION
   *
   * 0 for => Behler fc1
   * 1 for => Behler fc2
   * See DOI: 10.1002/qua.24890
   */
  void SetupCutoffFunction(int type, double cutoff_value);

  double get_cutoff();
  int get_cutoff_type();


  /*
   * The symmetric functions can be defined in the configuration file
   * This is in the JSON format. The keywords are defined in the top
   * of this header file. Parameters are passed as lists.
   */
  void LoadFromCFG(const char * cfg_filename);
  void SaveToCFG(const char * cfg_filename);

  // Print on stdout some info on the currently used symmetric functions
  void PrintInfo(void);

  /*
   * Returns the total number of symmetry functions used. This must match the number of
   * input layer of the atomic neural network.
   * It requires how many types there are in the network.
   */
  int GetTotalNSym(int N_types);
  int GetG2Sym(int N_types);
  int GetG4Sym(int N_types);

  // Get the type of the number of symmetric functions
  int get_n_g2();
  int get_n_g4();

};


/*
 * Here the symmetry functions
 */
double G2_symmetry_func(double cutoff, double eta, double RS, int cutoff_type,
			const double * coords, const int * atom_types, int N_atoms,
			int atom_index, int type_1);

double G4_symmetry_func(double cutoff, double zeta, double eta, int lambda, int cutoff_type,
			const double * coords, const int * atom_types, int N_atoms,
			int atom_index, int type_1, int type_2);

double cutoff_function(int cutoff_type, double r);


/*
 * The derivative of the G2 function with respect to an atom of index d_atm_index
 * and coordinate d_coord_index (0 = x, 1 = y, 2 = z)
 * 
 * All the other parameters match those of G2_symmetry_func
 */
double DG2_DX(double cutoff, double eta, double RS, int cutoff_type, 
  const double * coords, const int * atom_types, int N_atoms,
  int atom_index, int type_1, int d_atm_index, int d_coord_index);

// The same but for the G4 function
double DG4_DX(double cutoff, double zeta, double eta, int lambda, int cutoff_type,
			const double * coords, const int * atom_types, int N_atoms,
			int atom_index, int type_1, int type_2, int d_atm_index, int d_coord_index);

// Derivative of the cutoff function
double d_cutoff_function(int cutoff_type, double cutoff, double x);

  
#endif
