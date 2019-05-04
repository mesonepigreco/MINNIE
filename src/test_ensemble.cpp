/*
 * This test is used to check if the ensembles are properly read.
 * It can be used to perform some premilinar analysis in order to study better the symmetric functions to be used
 */

#include <iostream>
#include <string>
#include <libconfig.h++>
#include "ensemble.hpp"
#include "analyze_ensemble.hpp"

using namespace std;
using namespace libconfig;

void PrintUsage(void);

int main(int argc, char * argv[]) {
    // Load 
    Ensemble * ensemble = new Ensemble();

    // Check if one argument has been passed
    if (argc != 2) {
        PrintUsage();
        cerr << "Error, pass one argument with the configuration file" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Checking what to do..." << endl;

    if (!AnalyzeSymmetries(argv[1])) {
        cout << "Nothing to do" << endl;
        cout << "Try to setup an analysis in the configuration file." << endl;
    } else {
        cout << "Analysis performed." << endl;
    }

    return EXIT_SUCCESS;
}

void PrintUsage(void) {
    cout << "TEST ENSEMBLE" << endl;
    cout << "=============" << endl;
    cout << endl;

    cout << "Usage:" << endl;
    cout << "./test_ensemble.exe <config_file>" << endl;
    cout << endl;
    cout << "<config_file>   : A valid configuration file in JSON format"<< endl;
    cout << endl;
}
