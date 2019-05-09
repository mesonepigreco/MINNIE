#include "AtomicNetwork.hpp"
#include <math.h>

// A debugging flag
#define AN_DEB 0

double AtomicNetwork::GetEnergy(Atoms * coords, double * forces, int Nx, int Ny, int Nz, double ** grad_bias, double ** grad_sinapsis, double target_energy ) {
    int N_atms = coords->GetNAtoms();
    int N_sym = symm_f->GetTotalNSym(N_types);
    double* symm_fynctions = new double [N_sym * N_atms];
    double * first_layer;
    double ** tmp_biases, **tmp_sinapsis, *tmp_forces;
    double dS_dX, *dG_dX;
    double E_i, E_tot;
    double dumb = 1;
    int type;


    bool back_propagation = false;

    if ( ((bool)grad_bias) != ((bool)grad_sinapsis)) {
        cerr << "Error, both grad_bias and grad_sinapsis passed to GetEnergy must be consistent" << endl;
        cerr << "Or both allocated, or both NULL type" << endl;
        throw "";
    }
    if (forces || grad_bias) {
        back_propagation = true;
        tmp_forces = new double[N_lim* N_atms];


        // // Allocate all
        // if (grad_bias) {
        //     tmp_biases = new double* [coords->GetNTypes()];
        //     tmp_sinapsis = new double* [coords->GetNTypes()];
        //     for (int i = 0; i < coords->GetNTypes(); ++i) {
        //         tmp_biases[i] = new double [atomic_network.at(i)->get_nbiases()];
        //         tmp_sinapsis[i] = new double [atomic_network.at(i)->get_nsinapsis()];
        //     }
        // }
    }

    // Get the symmetric functions
    //cout << "Getting symmetric functions..." << endl;
    symm_f->GetSymmetricFunctions(coords, Nx, Ny, Nz, symm_fynctions);
    //cout << "Symmetric functions obtained." << endl;

    E_tot = 0;
    first_layer = new double[N_lim];
    for (int i = 0; i < N_atms; ++i) { // Parallelizable

        if (AN_DEB) {     
            for (int k = 0; k < N_sym; ++k) 
                printf("# SYM F [%d] of atom %d is : %.8e\n", k, i, symm_fynctions[N_sym*i + k]);
        }

        // Apply the PCA representation
        for (int j = 0; j < N_lim; ++j) {
            first_layer[j] = 0;
            for (int k = 0; k < N_sym; ++k) 
                first_layer[j] += eigvects[N_sym*j + k] * symm_fynctions[N_sym*i + k];
            first_layer[j] /= sqrt(eigvals[j]);

            if (AN_DEB) printf("# First layer [%d] of atom %d is : %.8e\n", j, i, first_layer[j]);
        }

        // Send into the neural network
        type = coords->types[i];
        GetNNFromElement(type)->PredictFeatures(1, first_layer, &E_i);
        E_tot += E_i;

        // If we need only the force, we can compute them
        if (back_propagation) {
            GetNNFromElement(type)->GetForces(tmp_forces + N_lim*i);
        }
    }

    // Backpropagation for the gradient
    // We need to rerun it, as the total energy is required to get the proper gradient of the sinapsis
    if (grad_bias) {
        for (int i = 0; i < N_atms; ++i) {
            type = coords->types[i];

            // Prepare the first layer once again
            for (int j = 0; j < N_lim; ++j) {
                first_layer[j] = 0;
                for (int k = 0; k < N_sym; ++k) 
                    first_layer[j] += eigvects[N_sym*j + k] * symm_fynctions[N_sym*i + k];
                first_layer[j] /= sqrt(eigvals[j]);
            }


            // Perform the Forward Propagation
            GetNNFromElement(type)->PredictFeatures(1, first_layer, &E_i);


            // Set the last node to the gradient of the loss function with respect to the atomic energy
            dumb = 2 * (E_tot - target_energy) / N_atms;

            // Backpropagate the last node to get the gradient of biases and sinapsis
            GetNNFromElement(type)->StepDescent(&dumb, grad_bias[type], grad_sinapsis[type], NULL);
        }
    }

    delete[] first_layer;


    // Get the forces
    if (back_propagation) {
/* 
        cout << "VARIANCES:" << endl;
        for (int i = 0; i < N_lim; ++i) 
            cout << eigvals[i] << "\t";
        cout << endl;
        cout << "EIGVECTS:" << endl;
        for (int i = 0; i < N_lim; ++i) {
            for (int j = 0; j < N_sym; ++j) {
                cout << eigvects[j*N_lim + i] << "\t" ;
            }
            cout << endl;
        } */

        // Print the tmp_forces
        /* cout << "Print tmp forces:" << endl;
        for (int i = 0; i < N_atms; ++i) {
            for (int j = 0; j < N_lim; ++j) 
                cout << tmp_forces[N_lim*i + j] << "\t";
            cout << endl;
        } */
        dG_dX = new double[N_sym * N_atms];

        for (int i = 0; i < N_atms; ++i) {
            for (int x = 0; x < 3; ++x) {
                forces[3*i + x] = 0;
                

                // Get the derivative of the symmetry function with respect to the atom
                symm_f->GetDerivatives(coords, Nx, Ny, Nz, i, x, dG_dX); 

                //cout << "Computing atom: " << i  << "/" << N_atms << " coord " << x ;
                
                // Now we must sum over all the symmetric functions
                // TODO: Restrict the sum only to nearby atoms
                // Atoms outside the cutoff are not going to have an effect

                //cout << "FORCE " << i << ", " << x << " BEFORE:" << forces[3*i +x] << endl;
                for (int j = 0; j < N_atms; ++j) {
                    for (int k = 0; k  < N_lim; ++k) {
                        dS_dX = 0;
                        for (int n = 0; n < N_sym; ++n) {
                           dS_dX += eigvects[N_lim*k + n] * dG_dX[N_sym * j + n];
                        }
                        dS_dX /= sqrt(eigvals[k]);

                        //cout << "AT " << j << "sqrteig:" << sqrt(eigvals[j]) << ", " << "dSdX:" << dS_dX << " ADDING:" << tmp_forces[N_lim*j + k] * dS_dX << endl;
                        // Get the force
                        forces[3*i + x] += tmp_forces[N_lim*j + k] * dS_dX;

                    }
                }
                //cout << "FORCE " << i << ", " << x << " AFTER:" << forces[3*i +x] << endl;


            }
        }
        //cout << "Done, deleting.." << endl;

        delete[] tmp_forces;
        delete[] dG_dX;
    }

    //cout << "Last deleting" << endl;


    // Free some memory
    delete[] symm_fynctions;

    //cout << "All deleted." << endl;

    return E_tot;
} 


// The saving function
void AtomicNetwork::SaveCFG(const char * PREFIX) {
    // Prepare the main cfg
    Config main_cfg;
    Setting &root = main_cfg.getRoot();

    // Add the types
    root.add(AN_ENVIRON, Setting::TypeGroup);
    Setting &AN = root[AN_ENVIRON];
    AN.add(AN_NTYPES, Setting::TypeInt) = N_types;
    Setting &types = AN.add(AN_TYPELIST, Setting::TypeArray);
    for (int i = 0; i < N_types; ++i) {
        types.add(Setting::TypeInt) = atomic_types.at(i);
    }

    int N_syms;
    N_syms = symm_f->GetTotalNSym(N_types);

    // Add the PCA
    root.add(AN_PCA, Setting::TypeGroup);
    Setting &PCA = root[AN_PCA];
    PCA.add(AN_NLIM, Setting::TypeInt) = N_lim;
    PCA.add(AN_NSYMS, Setting::TypeInt) = N_syms;
    Setting &sigma2 = PCA.add(AN_EIGVALS, Setting::TypeArray);
    for (int i = 0; i < N_lim; ++i) {
        sigma2.add(Setting::TypeFloat) = eigvals[i];
        Setting &ev = PCA.add(AN_EIGVECT + to_string(i), Setting::TypeArray);
        for (int j = 0; j < N_syms; ++j) {
            ev.add(Setting::TypeFloat) = eigvects[N_syms*i + j];
        }
    }


    // Add the symmetry functions
    // Write to the file the main
    string filename = string(PREFIX) + string("_main.cfg");
    main_cfg.writeFile(filename.c_str());    

    // Now save all the networks for each atomic type
    for (int i = 0; i < N_types; ++i) {
        filename = string(PREFIX) + string("_") + to_string( atomic_types.at(i)) + string(".cfg"); 
        atomic_network.at(i)->Save(filename);
    }

    // Save the symmetry functions
    filename = string(PREFIX) + string("_symfuncs.cfg");
    symm_f->SaveToCFG(filename.c_str());
}

void AtomicNetwork::LoadCFG(const char * PREFIX) {
    // Load the atomic network
    string filename;
    Config main_cfg;

    // Open the main cfg
    filename = string(PREFIX) + string("_main.cfg");
    try {
        main_cfg.readFile(filename.c_str());
    } catch (const FileIOException &e) {
        cerr << "Error, impossible to read file " << filename << endl;
        throw; 
    } catch (const ParseException &e) {
        cerr << "Syntax error in file " << filename << endl;
        cerr << "Line: " << e.getLine() << endl;
        cerr << e.getError() << endl;
        throw; 
    }

    if (! main_cfg.exists(AN_ENVIRON)) {
        cerr << "Error, file " << filename << " requires " AN_ENVIRON << endl;
        exit(EXIT_FAILURE);  
    }

    const Setting& root = main_cfg.getRoot();
    const Setting& AN = root[AN_ENVIRON];

    // Load the types of the atoms
    if (!AN.exists(AN_NTYPES)) {
        cerr << "Error, missing the " << AN_NTYPES << endl;
        cerr << "       from the enviroment " << AN_ENVIRON << endl;
        exit(EXIT_FAILURE);
    }

    try {
        AN.lookupValue(AN_NTYPES, N_types);
    } catch (...) {
        cerr << "Error while reading " << AN_NTYPES << ", please check carefully." << endl;
        throw;
    }

    if (!AN.exists(AN_TYPELIST)) {
        cerr << "Error, missing the keyword " << AN_TYPELIST << endl;
        cerr << "       inside the environment " << AN_ENVIRON << endl;
        exit(EXIT_FAILURE);
    }

    try {
        const Setting& cfg_types = AN[AN_TYPELIST];
        if (N_types != cfg_types.getLength()) {
            cerr << "Error, the number of types must match " << AN_NTYPES << endl;
            exit(EXIT_FAILURE);
        }

        // Load all the types
        for(int i = 0; i < N_types; ++i) 
            atomic_types.push_back(cfg_types[i]);
    } catch (const SettingException &e) {
        cerr << "Error within " << AN_TYPELIST <<", please check carefully" << endl;
        throw;
    }

    // Read the PCA info
    if (!main_cfg.exists(AN_PCA)) {
        cerr << "Error, environment " << AN_PCA << "not found" << endl;
        exit(EXIT_FAILURE); 
    }

    const Setting& pca = root[AN_PCA];
    int N_syms;
    try {
        N_lim = pca.lookup(AN_NLIM);
        N_syms = pca.lookup(AN_NSYMS);
    } catch (const SettingNotFoundException &e) {
        cerr << "Error, required keyword " << e.getPath() << endl;
        cerr << "       inside environment " << AN_PCA << endl;
        throw;
    } catch (const SettingException &e) {
        cerr << "Error with keyword " << e.getPath() << endl;
        throw;
    }

    eigvals = new double[N_lim];
    eigvects = new double[N_lim * N_syms];

    try {
        const Setting& cfg_vals = pca[AN_EIGVALS];
        
        for (int i = 0; i < N_lim; ++i) {
            eigvals[i] = cfg_vals[i];
            filename = string(AN_EIGVECT) + to_string(i);
            const Setting& cfg_v = pca[filename.c_str()];
            for (int j = 0; j < N_syms; ++j) {
                eigvects[N_lim*j + i] = cfg_v[j];
            }
        }
    } catch (const SettingNotFoundException &e) {
        cerr << "Error, keyword " << e.getPath() << " not found" << endl; 
        throw;
    } catch (const SettingTypeException &e) {
        cerr << "Error, keyword " << e.getPath() << " has wrong type" << endl;
        throw;
    } catch ( const SettingException &e) {
        cerr << "Error with keyword " << e.getPath() << ". Please, check carefully" << endl;
        throw;
    }

    // Read the Atomic networks
    NeuralNetwork* atomic_nn;
    for (int i = 0; i < N_types; ++i) {
        filename = string(PREFIX) + string("_") + to_string(atomic_types.at(i)) + string(".cfg");
        atomic_nn = new NeuralNetwork(filename);
        atomic_network.push_back(atomic_nn);
    }

    // Load the symmetric functions
    filename = string(PREFIX) + string("_symfuncs.cfg");
    symm_f = new SymmetricFunctions();
    symm_f->LoadFromCFG(filename.c_str());

    // Check if they are consistent
    int n_check = symm_f->GetTotalNSym(N_types);
    if (n_check != N_syms) {
        cerr << "Error, the number of symmetric functions in the file " << filename << endl;
        cerr << "       does not match what specified by parameter " << AN_NSYMS << endl;
        exit(EXIT_FAILURE);
    }
}


AtomicNetwork::~AtomicNetwork() {
    // Free memory
    delete[] eigvals;
    delete[] eigvects;

    atomic_network.clear();
    atomic_types.clear();

    delete symm_f;

}

AtomicNetwork::AtomicNetwork(const char * PREFIX) {
    // Load the network from file
    LoadCFG(PREFIX);
}

// Build the network from scratch
AtomicNetwork::AtomicNetwork(SymmetricFunctions* symf, Ensemble * ensemble, int Nlim, int Nhidden, int *nphl, int nhl) {
    // Check if the number of nodes per hidden layer
    if (!nphl && nhl <= 0) {
        cerr << "Error in construction of AtomicNetwork." << endl;
        cerr << "You must provide or the number of nodes per each hidden layer," << endl;
        cerr << "Or a valid number of nodes per layer." << endl;
        exit(EXIT_FAILURE);
    }

    // Setup the number of atomic types
    N_lim = Nlim;
    N_types = ensemble->GetNTyp();
    int N_sym = symf->GetTotalNSym(N_types);


    // Check if Nlim is bigger than N_syms
    if (N_lim > N_sym) {
        cerr << "Error, N_lim cannot exceed the number of symmetric functions " << N_sym << endl;
        cerr << "       Please, tidy your input." << endl;
        exit(EXIT_FAILURE);
    }

    // Setup the supercell
    //Nx = nx;
    //Ny = ny;
    //Nz = nz;

    // Perform the PCA
    double * means, *cvar_mat;
    GetCovarianceSymmetry(ensemble, symf, ensemble->N_x, ensemble->N_y, ensemble->N_z, means, cvar_mat);

    cout << "Nlim:" << N_lim << endl;
    cout << "Covariance matrix:" << endl;
    for (int i = 0; i < N_sym; ++i) {
        for (int j = 0; j < N_sym; ++j) {
            cout << cvar_mat[N_sym*i + j] << " ";
        } 
        cout << endl;
    }

    // Allocate eigenvalues and eigenvectors
    double * tmp_eigvals, *tmp_eigvects;
    tmp_eigvals = new double[N_sym];
    tmp_eigvects = new double[N_sym * N_sym];

    eigvals = new double[N_lim];
    eigvects = new double[N_lim * N_sym];

    // Perform the PCA
    Diagonalize(cvar_mat, N_sym, tmp_eigvals, tmp_eigvects);
    
    for (int i = 0; i < N_lim; ++i) {
        eigvals[i] = tmp_eigvals[i];
        for (int j = 0; j < N_sym; j++) {
            eigvects[N_lim*j + i] = tmp_eigvects[N_lim * j + i];
        }
    }
    delete[] tmp_eigvals;
    delete[] tmp_eigvects;

    symm_f = symf;

    // Get the total number of atomic types
    int partial_types = 0;
    Atoms * config;
    for (int i = 0; i < ensemble->GetNConfigs(); ++i) {
        ensemble->GetConfig(i, config);
        for (int j = 0; j < config->GetNAtoms(); ++j) {
            if (std::find(atomic_types.begin(), atomic_types.end(), config->types[j]) == atomic_types.end()) {
                atomic_types.push_back(config->types[j]);
                partial_types++;
                if (partial_types == N_types) 
                break;
            }
        }
        if (partial_types == N_types) break;
    }

    for (int i = 0; i < N_types; ++i) {
        NeuralNetwork* nn = new NeuralNetwork(N_lim, 1, Nhidden, nphl, nhl);
        atomic_network.push_back(nn);
    }
}

NeuralNetwork * AtomicNetwork::GetNNFromElement(int element_type) {
    for (int i = 0; i < N_types; ++i) {
        if (element_type == atomic_types.at(i)) {
            return atomic_network.at(i);
        }
    }
    cerr << "Error, element type " << element_type << " not found." << endl;
    exit(EXIT_FAILURE); 
}




double AtomicNetwork::GetLossGradient(Ensemble * training_set, double weight_energy, double weight_forces, double ** grad_biases, double ** grad_sinapsis, int offset, int n_configs) {
    // Check the parameters
    int n_conf = n_configs;
    if (offset >= training_set->GetNConfigs()) {
        cerr << "Error in function GetLossGradient: the offset " << offset << endl; 
        cerr << "cannot exceed the number of configurations " << training_set->GetNConfigs() << endl;
        cerr << "FILE: " << __FILE__ << " LINE: " << __LINE__ << endl;
        throw "Error"; 
    } else if (offset < 0) {
        cerr << "Error, the offset cannot be negative." << endl;
        cerr << "FILE: " << __FILE__ << " LINE: " << __LINE__ << endl; 
        throw "Error";
    }

    if (n_configs < 1) {
        n_conf = training_set->GetNConfigs();
    }

    if (n_conf + offset > training_set->GetNConfigs()) {
        cerr << "Error in function GetLossGradient: the selected number of configurations " << n_conf << " + " << offset << endl; 
        cerr << "cannot exceed the total number of configurations " << training_set->GetNConfigs() << endl;
        cerr << "FILE: " << __FILE__ << " LINE: " << __LINE__ << endl;
        throw "Error"; 
    } 


    double loss = 0;
    double energy = 0;
    double * forces = NULL;
    Atoms * config;
    for (int i = 0; i < n_conf; ++i) {
        // Get the current atomic configuration from the ensemble
        training_set->GetConfig(offset + i, config);

        forces = new double[config->GetNAtoms()*3];
        energy = training_set->GetEnergy(i);

        energy = GetEnergy(config, forces, training_set->N_x, training_set->N_y, training_set->N_z, grad_biases, grad_sinapsis, energy);

        // Get the loss function
        loss += weight_energy *(energy - training_set->GetEnergy(i) )*(energy - training_set->GetEnergy(i)) / (config->GetNAtoms());

        for (int j = 0; j < config->GetNAtoms() * 3; ++j) {
            loss += weight_forces * (forces[j] - training_set->GetForce(i, j/3, j%3))* (forces[j] - training_set->GetForce(i, j/3, j%3));
        }

        delete[] forces;
    }
    return loss;
}


void AtomicNetwork::TrainNetwork(Ensemble * training_set, string method, double step, int N_steps, bool use_lmin) {

    // Allocate the gradient biases and sinapsis for the whole atomic network 
    int n_networks = atomic_network.size();
    int n_biases_max = -1, n_sinapsis_max = -1;
    int tmp;

    double ** grad_biases = new double*[n_networks];
    double ** grad_sinapsis = new double *[n_networks];

    for (int i = 0; i < n_networks; ++i) {
        n_biases_max = atomic_network.at(i)->get_nbiases();
        n_sinapsis_max = atomic_network.at(i)->get_nsinapsis();
        grad_biases[i] = new double[n_biases_max];
        grad_sinapsis[i] = new double[n_sinapsis_max];
    }


    // Here the training section  
    double loss;
    double weight_energy, weight_force;

    double total_gradient = 0;

    // Check the minimization method
    if (method == AN_TRAINSD) {
        weight_energy = 1;
        weight_force = 0;
    }
    else if (method == AN_TRAINMC) {
        weight_energy = 0;
        weight_force = 1;
    } else {
        cerr << "Error, the ginven training method is unknown: " << method.c_str() << endl;
        cerr << "FILE:" << __FILE__ << " LINE: " << __LINE__ << endl;
        throw "";
    }

    for (int ka = 0; ka < N_steps; ++ka) {
        
        // Clean the gradient 
        for (int i = 0; i < n_networks; ++i) {
            for (int j = 0; j < atomic_network.at(i)->get_nbiases(); ++j)
                grad_biases[i][j] = 0;

            for (int j = 0; j < atomic_network.at(i)->get_nsinapsis(); ++j)
                grad_sinapsis[i][j] = 0;
        }

        loss = GetLossGradient(training_set, weight_energy, weight_force, grad_biases, grad_sinapsis);

        // Perform the optimization step
        if (method == AN_TRAINSD) {
            for (int i = 0; i < n_networks; ++i) {
                for (int j = 0; j < atomic_network.at(i)->get_nbiases(); ++j)  
                    atomic_network.at(i)->update_biases_value(j, - grad_biases[i][j] * step);
                
                for (int j = 0; j < atomic_network.at(i)->get_nsinapsis(); ++j)
                    atomic_network.at(i)->update_sinapsis_value(j, - grad_sinapsis[i][j] *step);
            }
        } else {
            cerr << "Method " << method.c_str() << " not yet implemented." << endl;
            exit(EXIT_FAILURE);
        }

        // Get the total gradient
        total_gradient = 0;
        for (int i = 0; i < n_networks; ++i) {
            for (int j = 0; j < atomic_network.at(i)->get_nbiases(); ++j)  
                total_gradient +=  grad_biases[i][j] * grad_biases[i][j];
            
            for (int j = 0; j < atomic_network.at(i)->get_nsinapsis(); ++j)
                total_gradient +=  grad_sinapsis[i][j] * grad_sinapsis[i][j];
        }

        // Print the info
        cout << " ===== STEP " << ka <<  " =====" << endl;
        cout << endl;
        cout << " Current Loss Function: " << scientific << loss << endl;
        cout << " Total gradient: " << total_gradient << endl;
        cout << endl;
        cout << fixed;
        
    }
}