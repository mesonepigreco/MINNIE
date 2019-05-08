#include "ensemble.hpp"
#include <string>
#include <fstream>
#include <libconfig.h++>

using namespace std;
using namespace libconfig;

Ensemble::Ensemble() {
    N_configs = 0;
}

Ensemble::~Ensemble() {
    // We can delete all the structures
    for (int i = 0; i < ensemble.size(); ++i) {
        delete ensemble.at(i);
    }

    for (int i = 0; i < forces.size();++i) {
        delete[] forces.at(i);
    }
    for (int i = 0; i < stresses.size(); ++i) {
        delete[] stresses.at(i);
    }
}


void Ensemble::Load(string folder_path, int N_conf, int population, int N_atoms, double alat) {
    // Load the ensemble from the specified path
    has_forces = false;
    has_stresses = false;
    N_configs = N_conf;

    // Check if the directory is followed by the '/' (add it in case)
    if (folder_path.back() != '/') {
        folder_path.append("/");
    }

    // Read the unit cell if any
    string filename = folder_path + "unit_cell_population" + to_string(population) + ".dat";
    fstream uc_file(filename.c_str());

    double unit_cell[9];
    bool has_unit_cell = false;
    if (uc_file) {
        // We found the unit cell

        for (int i = 0; i < 3; ++i) {
            if (! (uc_file >> unit_cell[3*i] >> unit_cell[3*i + 1] >> unit_cell[3*i + 2])) {
                cerr << "Error, the unit cell file is not in the right format!" << endl;
                throw "";
            }
        }
        has_unit_cell = true;
        uc_file.close();

        // Multiply the unit cell by the alat factor
        for (int i = 0; i < 9; ++i) 
            unit_cell[i] *= alat;
    }

    // Now start reading the ensemble
    for (int i = 0; i < N_configs; ++i) {
        // Open the file
        filename = folder_path + "scf_population" + to_string(population) + "_" + to_string(i+1) + ".dat";

        // Read the atomic configurations
        Atoms * new_config = new Atoms(N_atoms);
        new_config->ReadSCF(filename, alat);

        // If we read an unit cell override the one from the given atom
        if (has_unit_cell)
            for (int j = 0; j < 9; ++j)
                new_config->unit_cell[j] = unit_cell[j];

        // Push the atom in the list
        ensemble.push_back(new_config);

        // Check if the corresponding force exists
        filename = folder_path + "forces_population" + to_string(population) + "_" + to_string(i+1) + ".dat";
        uc_file.open(filename);
        if (uc_file) {
            if (i == 0) {
                has_forces = true;
            }
            if (i != 0 && !has_forces) {
                cerr << "Error, found a force file " << filename << " but nothing before!" << endl;
                throw "";
            }

            double x, y, z;
            double * force = new double[3*N_atoms];
            for (int j = 0; j < N_atoms; ++j) {
                if (! (uc_file >> x >> y >> z)) {
                    cerr << "Error while reading the forces " << filename << endl;
                    throw "";
                }
                force[3*j] = x;
                force[3*j + 1] = y;
                force[3*j + 2] = z;
            }
            forces.push_back(force);
            uc_file.close();
        } else {
            if (has_forces) {
                cerr << "Error, force file " << filename << " not found!" << endl;
                throw "";
            }
        }


        // Try to read also the stresses
        filename = folder_path + "pressures_population" + to_string(population) + "_" + to_string(i+1) + ".dat";
        uc_file.open(filename);
        if (uc_file) {
            if (i == 0) {
                has_stresses = true;
            }
            if (i != 0 && !has_stresses) {
                cerr << "Error, found a stress file " << filename << " but nothing before!" << endl;
                throw "";
            }
            double sx, sy, sz, syz, sxz, sxy, syx, szy, szx;
            double *stress = new double[6];
            if (! (uc_file >> sx >> sxy >> sxz >> syx >> sy >> syz >> szx >> szy >> sz)) {
                cerr << "Error while reading the stresses " << filename << endl;
                throw "";
            }
            stress[0] = sx;
            stress[1] = sy;
            stress[2] = sz;
            stress[3] = 0.5*(szy + syz);
            stress[4] = 0.5*(szx + sxz);
            stress[5] = 0.5*(sxy + syx);
            
            stresses.push_back(stress);
            uc_file.close();
        } else {
            if (has_stresses) {
                cerr << "Error, stress file " << filename << " not found!" << endl;
                throw "";
            }
        }
    }

    // Try to read the energy
    filename = folder_path + "energies_supercell_population" + to_string(population) + ".dat";
    uc_file.open(filename);
    if (uc_file) {

        double energy;
        for (int i = 0; i < N_conf; ++i) {
            if (! (uc_file >> energy)) {
                cerr << "Error while reading the energy file " << filename << endl;
                cerr << "Insufficient number of energies with respect to " << N_conf << endl;
                cerr << "FILE: " << __FILE__ << " LINE: " << __LINE__<< endl;
                throw "";
            }

            energies.push_back(energy);
        }
        uc_file.close();

    } else {
        if (has_forces) {
            cerr << "Error, found force but not energy: " << filename << " not found!" << endl;
            cerr << "FILE: " << __FILE__ << " LINE: " << __LINE__<< endl;
            throw "";
        }
    }

    // Update the number of configurations.
    GetNConfigs();


}

int Ensemble::GetNConfigs() {
    N_configs = ensemble.size();
    return N_configs;
}

void Ensemble::GetConfig(int index, Atoms* & structure) {
    structure = ensemble.at(index);
}

int Ensemble::GetNTyp(void) {
    int n_typs = 0, n_typs_tot = 0;
    Atoms * config;
    for (int i = 0; i < GetNConfigs(); ++i) {
        GetConfig(i, config);
        n_typs = config->GetNTypes();
        n_typs_tot =  (n_typs_tot < n_typs) ? n_typs : n_typs_tot; 
    }
    return n_typs_tot;
}


void Ensemble::LoadFromCFG(const char * config_file) {
    Config cfg;

    try {
        cfg.readFile(config_file);
    } catch (const FileIOException &e) {
        cerr << "FILE: " << __FILE__ << "LINE:" << __LINE__ << endl;
        cerr << "Error while reading the file " << config_file << endl;
        throw; 
    } catch (const ParseException &e) {
        cerr << "FILE: " << __FILE__ << "LINE:" << __LINE__ << endl;
        cerr << "Error while parsing the file: " << config_file << endl;
        cerr << "Line:  " << e.getLine() << endl;
        cerr << e.getError() << endl;
        throw;
    } 

    const Setting& root = cfg.getRoot();
    const Setting& ensemble_set = root[ENSEMBLE_ENVIRON];

    // Get the number of nodes
    int N_configs, pop = 1, N_atoms;
    double alat = 1;
    string data_dir;
    try {
        ensemble_set.lookupValue(ENSEMBLE_DATA, data_dir);
    } catch (...) {
        cerr << "Error while getting the ensemble data_dir" << endl;
        throw;
    }

    try {
        if (!ensemble_set.lookupValue(ENSEMBLE_NCONFIG, N_configs)) {
            cerr << "Error, specify the ensemble size" << endl;
            throw "";
        }
    } catch (...) {
        cerr << "Error while getting the ensemble size" << endl;
        throw;
    }

    try {
        if (!ensemble_set.lookupValue(ENSEMBLE_NATOMS, N_atoms)) {
            cerr << "Error, please specify the number of atoms" << endl;
            throw "";
        }
    } catch (...) {
        cerr << "Error with the number of atoms in the ensemble" << endl;
        throw;
    }

    try {
        ensemble_set.lookupValue(ENSEMBLE_POPULATION, pop);
        ensemble_set.lookupValue(ENSEMBLE_ALAT, alat);
    } catch (const SettingTypeException &e) {
        cerr << "Error while reading the configuration file" << endl;
        cerr << "Please, check the setting: " << e.getPath() << endl;
        cerr << "for correct type." << endl;
        throw;
    }

    // Now load the ensemble 
    Load(data_dir, N_configs, pop, N_atoms, alat);
}


double Ensemble::GetForce(int config_id, int atom_id, int coord_id) {
    if (coord_id < 0 || coord_id > 3) {
        cerr << "Error, the coord id must be between 0 and 3" << endl;
        throw "";
    }
    if (config_id < 0 || config_id > N_configs) {
        cerr << "Error, the number of configurations must be between 0 and " << N_configs << endl;
        throw "";
    }
    if (atom_id < 0 || atom_id > ensemble.at(config_id)->GetNAtoms()) {
        cerr << "Error, atom index " << atom_id << " not in the allowed range" << endl;
        throw "";
    }
    return forces.at(config_id)[3*atom_id + coord_id];
}


double Ensemble::GetEnergy(int config_id) {
    if (energies.size() == 0) {
        cerr << "Error, you cannot ask for energy. This ensemble does not have them." << endl;
        cerr << "FILE: " << __FILE__ << " LINE: " << __LINE__ << endl;
        throw "";
    }
    return energies.at(config_id);
}