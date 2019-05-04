/*
 * This code is meant to load symmetry functions and analyze them.
 * The ensemble is read and the symmetry functions are computed on the ensemble.
 * Then a simple analysis is conduced on the code.
 */

#ifndef HEADER_ANALYZE_ENSEMBLE
#define HEADER_ANALYZE_ENSEMBLE

#include <iostream>
#include <string>
#include <libconfig.h++>
#include "symmetric_functions.hpp"
#include "ensemble.hpp"

#define ANALENSEMBLE_ANALYSIS "analysis"
#define ANAL_PATH "save_path" // This is the keyword to save the results of the analysis
#define ANAL_PRINTSYM "printsym"  // This is used to print the symmetric functions for each atoms
#define ANAL_PRINTPCA "pca"
#define ANAL_PRINTFVSG "force-control"

using namespace std;
using namespace libconfig;

/*
 * This function load a configuration file with libconfig 
 * and perform the analysis, if requested.
 * 
 * It returns true if the analysis is performed, false otherwise.
 */
bool AnalyzeSymmetries(const char * config_file);

void PrintSymmetricFunctions(string file_path, SymmetricFunctions * symf, Ensemble * ens,
                            int Nx, int Ny, int Nz);


/*
 * Get the correlation between the i and j symmetric function in the ensemble.
 * Usefull to decide if a symmetric function is redundant
 */
//double GetCorrelation(double ** symmetric_functions, int N_sym, int N_atoms, int N_type, int N_configs, int sym_i, int sym_j);

/*
 * Get the covariance matrix between the symmetric functions. In this way it is possible
 * to perform a PCA to understand if there are redundant symmetric functions or not.
 * 
 * NOTE: The output array will be initialized
 * 
 * Parameters
 * ----------
 *      Ensemble : The atomic ensemble in which perform the analysis
 *      SymmetricFunctions : The symmetric functions
 *      Nx, Ny, Nz : The supercell size
 *  
 * Outputs
 * -------
 *      means : Array (size the total number of symmetric functions)
 *          The averages over all the atoms in the ensemble of the symmetric functions
 *      cvar_mat : Array (size the total number of symmetric functions to the square)
 *          The covariance matrix between the symmetric functions. <si sj> - <si><sj>
 */
void GetCovarianceSymmetry(Ensemble *, SymmetricFunctions*, int Nx, int Ny, int Nz, double * &means, double * &cvar_mat);


/*
 * For each couples of atoms in the whole ensemble, compare the modulus of the forces with
 * the difference of the symmetric functions.
 * In the ideal case, no event with a different modulus of the force but a similar symmetric function values
 * should occurr.
 * 
 * Parameters
 * ----------
 *      anal_path : string
 *          Path to the directory used to save the results.
 *      ensemble : The ensemble
 *      symm_func : The symmetric functions
 *      Nx, Ny, Nz : ints
 *          The size of the supercell in which the symmetric functions are evaluated.
 * 
 */
void AnalyzeForces(string anal_path, Ensemble * ensemble, SymmetricFunctions* symm_func, int Nx, int Ny, int Nz) ;
#endif