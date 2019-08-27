#ifndef ATOMS_CLASS
#define ATOMS_CLASS

using namespace std;


/*
 * Get the number of atoms in the given
 * scf file
 */
int GetNatmFromFile(string path) ;


class Atoms {
private:
  int N_atoms;
  int N_types;

public:
  int * types;
  double * coords;
  //double * forces;
  //double energy;
  //bool has_energy_force;

  // The unit cell
  double unit_cell[9];
  
  Atoms(int N);
  Atoms(string path_to_file);
  ~Atoms();

  // Return the number of atoms
  int GetNAtoms();

  // Print the coordinates in the standard output
  void PrintCoords();

  // Return the number of different types in the structure
  int GetNTypes();

  // Prepare the structure from an scf file
  // The optional alat parameters specifies the unit of measure with respect to Angstrom
  void ReadSCF(string path_to_file, double alat = 1);

  // Get atom coords in cartesian
  void GetAtomsCoords(int index, double &x, double &y, double &z);

  // Generate the Supercell (this is similar to the K_POINTS)
  /*
   * It will return a structure that, in the first N_atoms the coordinates of this atoms,
   * followed by the replica in the supercell. The arguments is how many times to repeat the
   * cell along each lattice versors.
   * Only odd values allowed. This is because if NX = 3, we will put one cell before and one after the
   * real one. This method helps to avoid surface effects.
   *  The target atoms must not been initialized.
   * 
   */
  void GenerateSupercell(int NX, int NY, int NZ, Atoms * &target);


  // Setup the energy and the force
  /*
   * Creates the force for the atom and copies forces. It sets up the has_energy_force variable
   * also.
   */
  //void SetEnergyForce(double energy, const double * force);

  int GetNAtomsPerType(int type);
};

#endif
