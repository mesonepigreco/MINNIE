#include <iostream>
#include <string>
#include "ParseInput.hpp"
#include "AtomicNetwork.hpp"
#include "ensemble.hpp"
#include "Trainer.hpp"
#include <stdlib.h>

#define N_ARGS 2
#define MODE_TRAIN "--train"
#define MODE_TEST "--test"
#define MODE_PREDICT "--predict"

using namespace std;
void PrintUsage(void) ;

int main (int argc, char * argv[]) {
    if (argc != N_ARGS + 1) {
        PrintUsage();
        return EXIT_FAILURE;
    }

    // Seed initialization  
    srand48(123);

    // Read the atomic network
    AtomicNetwork * network;
    bool initialize_params = GetNetworkFromInput(argv[1], network);


    // Pick the mode
    string mode(argv[2]);

    // Check the mode
    if (mode == MODE_TRAIN) {
        // Train the network
        cout << "MODE: TRAIN" << endl;

        Trainer * trainer = new Trainer(argv[1]);
        trainer->TrainAtomicNetwork(network, initialize_params);

        cout << "Train completed." << endl;
        cout << "Saving the results into PREFIX = " << trainer->save_prefix.c_str() << endl;

        network->SaveCFG(trainer->save_prefix.c_str());
        cout << "Done.";
    } else if (mode == MODE_TEST) {
        // Load the ensemble
        Ensemble * ensemble = new Ensemble();
        ensemble->LoadFromCFG(argv[1]);

        // Run the neural network for each configuration
        // And print energies in stdout (then)

        cout << "# Conf ID; Real Energy; Predicted energy" << endl;
        double pred_en;
        for (int i = 0; i < ensemble->GetNConfigs(); ++i) {
            // Get the atomic configuration
            Atoms * conf;
            ensemble->GetConfig(i, conf);

            pred_en = network->GetEnergy(conf, NULL, ensemble->N_x, ensemble->N_y, ensemble->N_z);

            cout << std::fixed << i << "\t" << std::scientific;
            cout << ensemble->GetEnergy(i) <<  "\t";
            cout << pred_en << endl;
        }
    } else {
        cerr << "Error, the mode " << mode.c_str() << " has still not been implemented." << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void PrintUsage(void) {
    cout << "Usage:" <<endl;
    cout << "MINNIE.exe configuration_file.cfg [Option]" << endl;
    cout << endl;
    cout << "You must specify as a first argument the configuration file for the current calculation." << endl;
    cout << "As a second argument what you want to do (train a network or predict features)." << endl;
    cout << endl << "Allowed options:" << endl;
    cout << "    " << MODE_TRAIN << "    => Train the network" << endl;
    cout << "    " << MODE_PREDICT<<"  => Predict the energy/forces of the ensemble" << endl;
    cout << endl;
}
