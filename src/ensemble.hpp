/*
 * In this file we define the ensemble as a collections of Atoms objects, through which
 * it is possible to train a Neural Network.
 * 
 * The ensemble can be loaded from a given data folder.
 * 
 */
#ifndef ENSEMBLE_HEADER
#define ENSEMBLE_HEADER

#include <string>
#include "atoms.hpp"
#include "symmetric_functions.hpp"
#include <libconfig.h++>

// Define the configuration keywords
#define ENSEMBLE_ENVIRON "Ensemble"
#define ENSEMBLE_DATA "data_dir"
#define ENSEMBLE_NCONFIG "n_configs"
#define ENSEMBLE_NATOMS "n_atoms"
#define ENSEMBLE_POPULATION "population"
#define ENSEMBLE_ALAT "alat"
#define ENSEMBLE_NX "N_sup_x"
#define ENSEMBLE_NY "N_sup_y"
#define ENSEMBLE_NZ "N_sup_z"

using namespace std;
using namespace libconfig;
class Ensemble {
private:
    int N_configs;

    vector<Atoms *> ensemble;
    vector<double> energies;
    vector<double *> forces;
    vector<double*> stresses; // Voight order

    bool has_forces;
    bool has_stresses;

public:

    // The standard supercell size for the ensemble
    int N_x, N_y, N_z;

    // Construct the ensemble
    Ensemble();

    // Destroy the ensemble
    ~Ensemble();

    // Load the ensemble from a folder
    /*
     * Inside the given folder the structures must be names as scf_population%d_%d.dat
     * Where the first decimal is the population id, while the second is the configuration index.
     * Optionally, if the unit cell is shared for all the configurations in the ensemble, an
     * extra file called unit_cell_population%d.dat can be placed in the folder directory, specifing the
     * unit cell in angstrom.
     * Note that the id of the configurations starts from 1. This is to mantain compatibility with
     * the Fortran code that generates the ensembles.
     * 
     * Parameters
     * ----------
     *      folder : string 
     *          Path to the folder containing the ensemble
     *      N_configs : int
     *          Number of configurations inside the ensemble
     *      population : int
     *          ID of the ensemble inside the folder
     *      N_atoms : int
     *          The number of atoms for each configuration in the ensemble.
     *      alat : double
     *          The units [in Angstrom] in which the atomic position are written. By default it is angstrom.
     */
    void Load(string folder, int N_configs, int population, int N_atoms, double alat = 1);


    /*
     * Load the ensemble from a cinfiguration file.
     * The keyword for the configuration files are defined in the top
     * of this header.
     */
    void LoadFromCFG(const char * config_file);

    // Interact with the ensemble
    int GetNConfigs(void);
    int GetNTyp(void); // Get the total number of atomic species

    // Get the atoms configuration at a given index
    // config is a pointer to the configuration that will be setted to point to the desidered Atoms object.
    void GetConfig(int index, Atoms* &config);

    //Get the force
    double GetForce(int config_id, int atom_id, int coord_id);

    // Get the energy
    double GetEnergy(int config_id);
};
#endif