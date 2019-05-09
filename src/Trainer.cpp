#include "Trainer.hpp"
#include <libconfig.h++>

using namespace std; 
using namespace libconfig;

Trainer::Trainer() {
    method = AN_TRAINSD;
    step = 0.01;
    N_steps = 1000;
    use_lmin = false;

    training_set = new Ensemble();
}

Trainer::Trainer(const char * file) {
    Trainer();
    SetupFromCFG(file);
}

Trainer::~Trainer() {
    delete training_set;
}

void Trainer::SetupFromCFG(const char * file) {
    Config main;

    try {
        main.readFile(file);
    } catch (const FileIOException & e) {
        cerr << "IO Error while opening file: " << file << endl;
        cerr << e.what() << endl;
        throw;
    } catch (const ParseException& e) {
        cerr << "Syntax error in the configuration file " << file << endl;
        cerr << "FILE: " << e.getFile() << "LINE: " << e.getLine() << endl;
        cerr << "Error: " << e.getError() << endl;
        cerr << e.what() << endl;
        throw;
    }

    // Look for the Training environment
    if (!main.exists(TRAINER_KEYWORD)) {
        cerr << "Error, the '" << TRAINER_KEYWORD << "' is required in the file " << file << endl;
        cerr << "SOURCE FILE: " << __FILE__ << " (" << __LINE__ << ")"<<endl;
        throw "";
    }

    // Setup the ensemble
    training_set->LoadFromCFG(file);


    const Setting& root = main.getRoot();
    const Setting& train_env = root[TRAINER_KEYWORD];

    // Read all the variables
    try {
        train_env.lookupValue(TRAINER_METHOD, method);
        step = train_env.lookup(TRAINER_STEP);
        N_steps = train_env.lookup(TRAINER_NITERS);
        train_env.lookupValue(TRAINER_USE_LMIN, use_lmin);
    } catch ( const SettingNotFoundException &e) {
        cerr << "Error, the namelist: " << TRAINER_KEYWORD << " requires key: " << e.getPath() << endl;
        cerr << e.what()<<endl;
        throw;
    } catch (const SettingTypeException &e) {
        cerr << "Error, setting: " << e.getPath() << " has the wrong type." << endl;
        cerr << e.what() << endl;
        throw;
    } catch (const SettingNameException &e) {
        cerr << "Error, wrong name of the setting: " << e.getPath() << endl;
        cerr << e.what() << endl;
        throw;
    }
}


void Trainer::TrainAtomicNetwork(AtomicNetwork* target, bool precondition) {
    // Setup the final biases to reproduce the average energy
    if (precondition) {
        int n_typ = target->N_types;
        double * av_energy_per_type = new double[n_typ]();
        double * av_energy2_per_type = new double[n_typ]();
        Atoms * config;

        for (int i = 0; i < training_set->GetNConfigs(); ++i) {
            double energy = training_set->GetEnergy(i);
            
            training_set->GetConfig(i, config);

            for (int j = 0; j < n_typ; ++j) {
                int n_of_type = 0;

                // Count how many atoms of this type
                for (int k = 0; k < config->GetNAtoms(); ++k) 
                    if (config->types[k] == j)
                        n_of_type++;

                double e_net;
                e_net =  energy * n_of_type / (double) config->GetNAtoms();
                av_energy_per_type[i] += e_net / training_set->GetNConfigs();
                av_energy2_per_type[i] += e_net * e_net/ training_set->GetNConfigs();
            }
            
        }

        NeuralNetwork * network;
        double sigma;
        int n_last;
        for (int j = 0; j < n_typ; ++j) {
            // Setup the last bias network
            network = target->GetNNFromElement(j);

            network->set_biases_value(network->get_nbiases() - 1, av_energy_per_type[j]);

            sigma = av_energy2_per_type[j] - av_energy_per_type[j] * av_energy_per_type[j];
            n_last = network->N_nodes.at(network->N_hidden_layers);

            sigma = sqrt(sigma/n_last);        

            // Setup the last sinapsis
            for (int k = 0; k < n_last;++k) {
                network->set_sinapsis_value( network->get_nsinapsis() - k, sigma);
            }
        }
    }

    target->TrainNetwork(training_set, training_set->N_x, training_set->N_y, training_set->N_z,
        method, step, N_steps, use_lmin);
}