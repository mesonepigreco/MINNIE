#include <iostream>
#include <string>
#include "ParseInput.hpp"
#include "AtomicNetwork.hpp"
#include "ensemble.hpp"
#include "Trainer.hpp"

#define N_ARGS 2
#define MODE_TRAIN "--train"
#define MODE_PREDICT "--predict"

using namespace std;
void PrintUsage(void) ;

int main (int argc, char * argv[]) {
    if (argc != N_ARGS + 1) {
        PrintUsage();
        return EXIT_FAILURE;
    }

    // Read the atomic network
    AtomicNetwork * network;
    GetNetworkFromInput(argv[1], network);




    // Pick the mode
    string mode(argv[2]);

    // Check the mode
    if (mode == MODE_TRAIN) {
        // Train the network
        cout << "MODE: TRAIN" << endl;

        Trainer * trainer = new Trainer(argv[1]);
        trainer->TrainAtomicNetwork(network);

        cout << "Train completed." << endl;
        cout << "Saving the results into PREFIX = " << trainer->save_prefix.c_str() << endl;

        network->SaveCFG(trainer->save_prefix.c_str());
        cout << "Done.";
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