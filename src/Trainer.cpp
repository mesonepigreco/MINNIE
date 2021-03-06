#include "Trainer.hpp"
#include <libconfig.h++>
#include <math.h>
#include "utils.hpp"

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
    method = AN_TRAINSD;
    step = 0.01;
    N_steps = 1000;
    use_lmin = false;

    training_set = new Ensemble();

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
    //training_set = new Ensemble();
    training_set->LoadFromCFG(file);


    const Setting& root = main.getRoot();
    const Setting& train_env = root[TRAINER_KEYWORD];

    // Read all the variables
    try {
        train_env.lookupValue(TRAINER_METHOD, method);
        step = train_env.lookup(TRAINER_STEP);
        N_steps = train_env.lookup(TRAINER_NITERS);
        if (train_env.exists(TRAINER_TEMP))
            temperature = train_env.lookup(TRAINER_TEMP);
        train_env.lookupValue(TRAINER_USE_LMIN, use_lmin);
        if (! train_env.lookupValue(TRAINER_SAVE_PREFIX, save_prefix)) {
            cerr << "Error, you need to specify the " << TRAINER_SAVE_PREFIX <<" inside the " << TRAINER_KEYWORD << " environ." << endl;
            throw "";
        }
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
        double av_energy_per_type;
        double av_energy2_per_type;
        Atoms * config;

        for (int i = 0; i < training_set->GetNConfigs(); ++i) {
            double energy = training_set->GetEnergy(i);
            
            training_set->GetConfig(i, config);

	    double e_net;
	    // Get the fraction of the total energy by this atomic type
	    e_net =  energy  / (double) config->GetNAtoms();
	    av_energy_per_type += e_net / training_set->GetNConfigs();
	    av_energy2_per_type += e_net * e_net/ training_set->GetNConfigs();
	}
        double sigma;
        NeuralNetwork * network;
        int n_last;

    sigma = 1;
	//sigma = av_energy2_per_type - av_energy_per_type*av_energy_per_type;
	//sigma = sqrt(sigma);
	
	//cout << "Average energy per type: " << av_energy_per_type << endl;
	//cout << "Sigma energy per type: " << sigma << endl;

        for (int j = 0; j < n_typ; ++j) {
            // Setup the last bias network
            network = target->GetNNFromElement(j);

            // All the biases to zero
            for (int k = 0; k < network->get_nbiases(); ++k) 
                network->set_biases_value(k, 0);


            // All the sinapsis must be initialized using the Xavier (Glorot) scheme:
            /*
             *
             * W ~ Normal ( sigma = sqrt(2) / sqrt(n_in + n_out))
             * 
             * Where n_in are the number  of input neurons and n_out are the number of output neurons.
             */
            int current_node = 0;
            int shift = 0;
            for (int k = 0; k < network->get_nsinapsis(); ++k)  {
                int N_in = network->N_nodes.at(network->get_sinapsis_starting_layer(k));
                int N_out = network->N_nodes.at(network->get_sinapsis_starting_layer(k) + 1);
                sigma = sqrt(2) / sqrt(N_in + N_out);
                cout << "Synapsis " << k << " | Layer from " << N_in << " to " << N_out << endl;
                network->set_sinapsis_value(k, random_normal(0, sigma));
            }
        }
        // Print all the biases
        cout << "BIASES:" << endl;
        for (int i = 0; i < network->get_nbiases(); ++i) 
            cout << "bias(" << i << ") = " << network->get_biases_value(i) << endl;

        // Print all the biases
        cout << "SINAPSIS:" << endl;
        for (int i = 0; i < network->get_nsinapsis(); ++i) 
            cout << "sinapsis(" << i << ") = " << network->get_sinapsis_value(i) << endl;

        


        // Test the energy prediction
        training_set->GetConfig(0, config);
        double energy_predicted = target->GetEnergy(config);
        cout << endl;
        cout << "N atoms: " << config->GetNAtoms() << endl;
        cout << "Predicted energy: " << energy_predicted << endl;
        cout << "Real energy: " << training_set->GetEnergy(0) << endl;

        cout << endl << "Input layer (after sym functions)" << endl;
        for (int i = 0; i < network->N_nodes.at(0); ++i) {
            cout << i << ")  neuron = " << network->get_neuron_value(0,i) << endl;
        }

        cout << endl << "Last layer neurons:" << endl;
        for (int i = 0; i < network->N_nodes.at(network->N_hidden_layers); ++i) {
            double sinp;
            sinp = network->get_sinapsis_value( network->get_sinapsis_index(network->N_hidden_layers+1, i,0)) ;
            cout << i << ") " <<  network->get_neuron_value(network->N_hidden_layers, i);
            cout << "  " << sinp << " index = " << network->get_sinapsis_index(network->N_hidden_layers, i,0)<<  endl;
        }

        cout << "The energy layer:" << endl;
        cout << "Last" << ") " <<  network->get_neuron_value(network->N_hidden_layers + 1, 0) << endl;
    }

    target->TrainNetwork(training_set, method, step, N_steps, use_lmin, temperature);
}
