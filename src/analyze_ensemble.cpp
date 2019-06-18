#include "analyze_ensemble.hpp"
#include "ensemble.hpp"
#include "symmetric_functions.hpp"
#include "matrix.h"
#include <fstream>
#include <iomanip>
#include <math.h>
#include <boost/filesystem.hpp>

using namespace boost;

#define DEB_ANALENS 1

bool AnalyzeSymmetries(const char * config_file) {
    Config cfg;
    // Try to open the configuration file
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

    // Check if some analysis is requested
    bool is_analysis = cfg.exists(ANALENSEMBLE_ANALYSIS);

    // If no analysis is requested, exit from the method
    if (! is_analysis) return false;

    // Prepare the ensemble and the symmetry functions
    SymmetricFunctions * sym_functs = new SymmetricFunctions();
    Ensemble * ensemble = new Ensemble();

    // Load the ensemble and the symmetric functions from the CFG file
    sym_functs->LoadFromCFG(config_file);
    cout << "Symmetric functions loaded." << endl;
    sym_functs->PrintInfo();

    ensemble->LoadFromCFG(config_file);
    cout << "Ensemble loaded." << endl;

    // If the analysis exists check the type
    string anal_type, anal_path;
    try{
        cfg.lookupValue(ANALENSEMBLE_ANALYSIS, anal_type);
    } catch (...) {
        cerr << "Error, please check carefully the " << ANALENSEMBLE_ANALYSIS << " keyword." << endl;
        cerr << "(It should be a string)" << endl;
        throw;
    }

    // Get the file path
    if (!cfg.exists(ANAL_PATH)) {
        cerr << "Error, please specify the path in which you want to save analisys results" << endl;
        cerr << "KEYWORD: " << ANAL_PATH << endl;
        throw;
    }
    try {
        cfg.lookupValue(ANAL_PATH, anal_path);
    } catch (...) {
        cerr << "Error, please check carefully the " << ANAL_PATH << " keyword." << endl;
        cerr << "(It should be a string)" << endl;
        throw;
    }

    // Create the directory if it does not exists
    bool dir_exists, dir_is_dir;
    dir_exists = filesystem::exists(anal_path);
    dir_is_dir = filesystem::is_directory(anal_path);
    if (dir_exists && !dir_is_dir) {
        cerr << "Error, the specified " << ANAL_PATH << " exists, and it is not a directory." << endl;
        cerr << "Aborting." << endl;
        throw "";
    }
    if (!dir_exists) {
        dir_is_dir = filesystem::create_directory(anal_path);
        if (!dir_is_dir) {
            cerr << "Error while creating the directory " << ANAL_PATH << endl;
            cerr << "Aborting." << endl;
            throw "";
        }
    }

    // Get the supercell size
    int Nx = 1, Ny = 1, Nz = 1;
    try{
        cfg.lookupValue(SUPERCELL_NX, Nx);
        cfg.lookupValue(SUPERCELL_NY, Ny);
        cfg.lookupValue(SUPERCELL_NZ, Nz);
    } catch (const SettingTypeException &e) {
        cerr << "Error while setting the supercell:" << endl;
        cerr << "Wrong type of " << e.getPath() << endl;
        throw;
    }

    cout << "Starting the analysis..." << endl;
    if (anal_type == ANAL_PRINTSYM) {
        PrintSymmetricFunctions(anal_path, sym_functs, ensemble, Nx, Ny, Nz);
    } else if (anal_type == ANAL_PRINTPCA) {    
        // Perform the PCA on the ensemble
        double *means, *cvar_mat;
        double *eigvals, *eigvects;
        int N_tot;

        // Compute the covariance
        cout << "Getting the covariance matrix..." << endl;
        GetCovarianceSymmetry(ensemble, sym_functs, Nx, Ny, Nz, means, cvar_mat);
        cout << "Covariance matrix obtained" << endl;
        N_tot = sym_functs->GetTotalNSym(ensemble->GetNTyp());

        
        // Save the matrix
        ofstream cov_mat_s;
        cov_mat_s.open(anal_path + "/covariance_matrix.dat");
        if (!cov_mat_s) {
            cerr << "Error while opening the " << anal_path.c_str() << "/covariance_matrix.dat;" << endl;
            throw "";
        }
        cov_mat_s << "# Covariance matrix" << endl;
        cov_mat_s << setprecision(8) << fixed << scientific;
        for (int i = 0; i  < N_tot; ++i) {
            for (int j = 0; j < N_tot; ++j) 
                cov_mat_s << "\t" << cvar_mat[N_tot*i + j];
            cov_mat_s << endl;
        }
        cov_mat_s.close();
        cout << "Covariance matrix of symmetric functions saved in " << anal_path.c_str() << "/covariance_matrix.dat;" << endl;

        
        // Save the observables
        cov_mat_s.open(anal_path + "/average.dat");
        if (!cov_mat_s) {
            cerr << "Error while opening the " << anal_path.c_str() << "/average.dat;" << endl;
            throw "";
        }
        cov_mat_s << "# Average and standard dev of the symmetric function values" << endl;
        cov_mat_s << "# Sym func ID; Mean; Stdev" << endl;
        cov_mat_s << setprecision(8) << fixed << scientific;
        for (int i = 0; i  < N_tot; ++i) {
            cov_mat_s << i << "\t" << means[i] << "\t"<< sqrt(cvar_mat[N_tot*i + i]) << endl;
        }
        cov_mat_s.close();
        cout << "Average symmetric functions saved in " << anal_path.c_str() << "/average.dat;" << endl;

        // Diagonalize the result
        eigvals = new double[N_tot];
        eigvects = new double[N_tot*N_tot];
        Diagonalize(cvar_mat, N_tot, eigvals, eigvects);

        // Save the eigenvectors
        cov_mat_s.open(anal_path + "/eigvects.dat");
        if (!cov_mat_s) {
            cerr << "Error while opening "<< anal_path.c_str() << "/eigvects.dat;" << endl;
            throw "";
        }
        cov_mat_s << "# Eigen vectors (columns)" << endl;
        cov_mat_s << setprecision(8) << fixed << scientific;
        for (int i = 0; i  < N_tot; ++i) {
            for (int j = 0; j < N_tot; ++j) 
                cov_mat_s << "\t" << eigvects[N_tot*i + j];
            cov_mat_s << endl;
        }
        cov_mat_s.close();
        cout << "Eigenvectors saved in " << anal_path.c_str() << "/eigvects.dat;" << endl;

        // Save the eigen values
        cov_mat_s.open(anal_path + "/eigvals.dat");
        if (!cov_mat_s) {
            cerr << "Error while opening "<< anal_path.c_str() << "/eigvals.dat;" << endl;
            throw "";
        }
        cov_mat_s << "# Eigenvalues" << endl;
        cov_mat_s << setprecision(8) << fixed << scientific;
        for (int i = 0; i  < N_tot; ++i) {
            cov_mat_s << eigvals[i] << endl;
        }
        cov_mat_s.close();
        cout << "Eigenvalues saved in " << anal_path.c_str() << "/eigvals.dat;" << endl;
        cout << endl;

        // Free memory
        cout << "Freeing memory" << endl;
        delete[] eigvals;
        delete[] eigvects;
        delete[] cvar_mat;
        delete[] means;
        cout << "Done." << endl;
        
    } else if (anal_type == ANAL_PRINTFVSG) {
        // Perform the force controll
        AnalyzeForces(anal_path, ensemble, sym_functs, Nx, Ny, Nz);
    }
    else {
        cerr << "Error, type analysis " << anal_type.c_str() << " unrecognized." << endl;
        throw invalid_argument("Invalid analysis type.");
    }

    return true;
}

void PrintSymmetricFunctions(string file_path, SymmetricFunctions * symf, Ensemble * ens,
                            int Nx, int Ny, int Nz) {
    ofstream of(file_path + "/symmetric_functions.dat");
    double * sym_values;
    if (!of) {
        cerr << "Error while opening " << file_path.c_str() << endl;
        cerr << "Please, verify that you can write in that position." << endl;
        throw FileIOException();
    }

    // Print decimal with 8 positions
    of << fixed << setprecision(8);

    Atoms * atm;
    int N_syms;
    for (int i = 0; i < ens->GetNConfigs(); ++i) {
        ens->GetConfig(i, atm);

        // Allocate the correct number of symmetric functions
        N_syms = atm->GetNAtoms() * symf->GetTotalNSym(atm->GetNTypes());
        sym_values = new double[N_syms];

        // Get the symmetric functions for the given structure
        symf->GetSymmetricFunctions(atm, Nx, Ny, Nz, sym_values); 

        // Print the values of the symmetric functions in the file
        of << i;
        for (int j = 0; j < N_syms; ++j) 
            of << "\t" << sym_values[j];
        of << endl;

        // Free the memory
        delete[] sym_values;
    }
    of.close();
}
// TO BE CORRECTED, THE NUMBER OF ATOMS MAY CHANGE IN THE ENSEMBLE
// THEREFORE THIS FUNCTION SHOULD TAKE THE WHOLE ENSEMBLE 
/* 
double GetCorrelation(double ** symmetric_functions, int N_sym, int N_atoms, int N_type, int N_configs, int sym_i, int sym_j) {
    double cvar2 = 0, var2i = 0, var2j = 0;
    double mean_i = 0, mean_j = 0;
    for (int i = 0; i < N_type; ++i) {
        for (int j = 0; j < N_atoms; ++j) {
            for (int k = 0; k < N_configs; ++k) { // WRONG
                cvar2 += symmetric_functions[k][sym_i + N_sym * i + N_sym * N_atoms] * 
                    symmetric_functions[k][sym_j + N_sym * i + N_sym * N_atoms];
                var2i += symmetric_functions[k][sym_i + N_sym * i + N_sym * N_atoms] * 
                    symmetric_functions[k][sym_i + N_sym * i + N_sym * N_atoms];
                var2j += symmetric_functions[k][sym_j + N_sym * i + N_sym * N_atoms] * 
                    symmetric_functions[k][sym_j + N_sym * i + N_sym * N_atoms];
                mean_i += symmetric_functions[k][sym_i + N_sym * i + N_sym * N_atoms];
                mean_j += symmetric_functions[k][sym_j + N_sym * i + N_sym * N_atoms];
            }
        }
    }
    cvar2 /= N_type * N_configs * N_atoms;
    mean_i /= N_type * N_configs * N_atoms;
    mean_j /= N_type * N_configs * N_atoms;

    double sigma_ij = cvar2 - mean_i*mean_j;
    double sigma_i = var2i - mean_i*mean_i;
    double sigma_j = var2j - mean_j*mean_j;
    return (sigma_ij / sqrt(sigma_i * sigma_j));
}

bool AddG2Function(SymmetricFunctions * symf, Ensemble * ensemble, int Nx, int Ny, int Nz, double eta_start, double eta_step, double eta_max, double max_corr) {
    double Rs = 0;
    double eta = eta_start;

    Atoms * structure;
    double ** symm_functions = (double**) malloc(sizeof(double*) * ensemble->GetNConfigs());
    double max_corr = 0;
    double tmp_cor;
    int N_typ = 0;

    while (eta < eta_max) {
        // Add the trial function
        symf->AddG2Function(0, eta);

        // Allocate the symmetric functions
        for (int i = 0; i < ensemble->GetNConfigs(); ++i) {
            // Get the atomic structure
            ensemble->GetConfig(i, structure);

            // Compute the symmetric functions
            N_typ = structure->GetNTypes();
            symm_functions[i] = (double*) calloc(sizeof(double), symf->GetTotalNSym(N_typ));
            symf->GetSymmetricFunctions(structure, Nx, Ny, Nz, symm_functions[i]);
        }

        // Get the correlation
        eta += eta_step;
    }
}
 */


void GetCovarianceSymmetry(Ensemble * ens, SymmetricFunctions* symmf, int Nx, int Ny, int Nz, 
                            double * &means, double * &cov_mat) {
    int nat_tot = 0;
    int ntyp = ens->GetNTyp();
    int N_sym_tot = symmf->GetTotalNSym(ntyp);
    double * sym_values;
    int N_atoms;
    means = new double[N_sym_tot];
    cov_mat = new double[N_sym_tot * N_sym_tot];


    // Prepare the means and covariance matrix

    // Perform a zero initialization
    for (int i = 0; i < N_sym_tot; ++i) means[i] = 0;
    for (int i = 0; i < N_sym_tot * N_sym_tot; ++i) cov_mat[i] = 0;

    Atoms * config;
    for (int k = 0; k < ens->GetNConfigs(); ++k) {
        // Get the atomic configuration
        ens->GetConfig(k, config);

        //cout << "Config number " << k << endl;
        //cout << "Ntyp:" << ntyp << " vs " << config->GetNTypes() << endl;

        // Print info
        //config->PrintCoords();
        N_atoms = config->GetNAtoms();

        // Get the symmetric functions for this configuration
        sym_values = new double[N_sym_tot*N_atoms];
        symmf->GetSymmetricFunctions(config, Nx, Ny, Nz, sym_values);


        // Cycle over the atoms
        for (int h = 0; h < N_atoms; ++h) {
            // Setup the symmetric functions
            //cout << "Getting symmetries" << endl;
            //cout << "Done" << endl;

            //cout << "Atom h = " << h << " / " << config->GetNAtoms() << endl;

            // Add the atomic count
            nat_tot++;

            // Cycle over the symmetric functions
            for (int i = 0; i < N_sym_tot; ++i) {
                // Get the mean value

                //cout << "Sym value (" << i << ") is " << sym_values[i] << endl;
                //cout << "i = " << i << " / " << N_sym_tot << endl;
                means[i] += sym_values[i + N_sym_tot * h];


                // Get the mean square
                for (int j = 0; j < N_sym_tot; ++j) {
                    //cout << "j = " << j << " / " << N_sym_tot << "   cmat: " << N_sym_tot*i + j << " / " << N_sym_tot*N_sym_tot << endl;

                    cov_mat[N_sym_tot*i + j] += sym_values[i + N_sym_tot * h] * sym_values[j + N_sym_tot * h];
                }
            }
        }

        delete[] sym_values;
    }
    cout << "Averages computed." << endl;

    // Divide by the total number of atoms to get the real averages
    
    for (int i = 0; i < N_sym_tot; ++i) 
        means[i] /= nat_tot;
        
    for(int i = 0; i < N_sym_tot*N_sym_tot; ++i)
        cov_mat[i] /= nat_tot;

    // Get the real covariance matrix by subtracting the one body averages
    ofstream cov_file;
    if (DEB_ANALENS) {
        cov_file.open("CovarianceMatrix.dat");
        cov_file << std::scientific;
    }
    for (int i = 0; i < N_sym_tot; ++i) {
        for (int j = 0; j< N_sym_tot; ++j) {
            cov_mat[N_sym_tot*i + j] -= means[i] * means[j];
            if (DEB_ANALENS)
                cov_file << cov_mat[N_sym_tot * i + j] << " ";
        }
        if (DEB_ANALENS)
            cov_file << endl;
    }
    if (DEB_ANALENS)
        cov_file.close();
}


void AnalyzeForces(string anal_path, Ensemble * ensemble, SymmetricFunctions* symm_func, int Nx, int Ny, int Nz) {
    // Get all the symmetric functions

    double ** total_sym_functions;
    int N_configs;
    int N_atoms, nat_j;
    int N_types;
    int N_sym;
    Atoms * config;

    N_types = ensemble->GetNTyp();
    N_configs = ensemble->GetNConfigs();

    N_sym = symm_func->GetTotalNSym(N_types);

    total_sym_functions = new double* [N_configs];

    double g_dist;
    double f_dist;

    ofstream file;
    file.open(anal_path + "/force_vs_symm.dat");
    if (!file) {
        cerr << "Error while creating " << anal_path.c_str() << "/force_vs_symm.dat" << endl;
        cerr << "Aborting." << endl;
        throw "";
    }

    file << "# Symm func dist; Force; config1; atom1; config2; atom2" << endl;
    file << setprecision(8) << scientific;

    // Compute the symmetric functions
    for (int i = 0; i < N_configs; ++i) {
        // Get the atomic configuration
        ensemble->GetConfig(i, config);

        N_atoms = config->GetNAtoms();


        total_sym_functions[i] = new double[N_sym * N_atoms];
        symm_func->GetSymmetricFunctions(config, Nx, Ny, Nz, total_sym_functions[i]);

        for (int j = 0; j <= i; ++j) {
            nat_j = config->GetNAtoms();

            for (int h = 0; h < N_atoms; ++h) {
                for (int k = 0; k < nat_j; ++k) {
                    if (h == k && i == j) continue;

                    // Get the distance between the configurations
                    g_dist = 0;
                    f_dist = 0;
                    for (int n = 0; n < N_sym; ++n) 
                        g_dist += (total_sym_functions[i][N_sym*h + n] - total_sym_functions[j][N_sym*k + n]) * 
                            (total_sym_functions[i][N_sym*h + n] - total_sym_functions[j][N_sym*k + n]);
                    g_dist = sqrt(g_dist);
                    
                    for (int n = 0; n < 3; ++n) 
                        f_dist += ensemble->GetForce(i, h, n) *ensemble->GetForce(i, h, n)  - 
                            ensemble->GetForce(j, k, n) * ensemble->GetForce(j, k, n);
                    f_dist = sqrt(fabs(f_dist));

                    file << g_dist << "\t" << f_dist << "\t" << i << "\t" << h << "\t" << j << "\t" << k << endl; 
                }
            }
        }
    }

    // Free memory
    for (int i = 0; i < N_configs; ++i) delete[] total_sym_functions[i];
    delete[] total_sym_functions;

    file.close();
}