#include <iostream>
#include <string>

#include <libconfig.h++>

#include "AtomicNetwork.hpp"
#include "ParseInput.hpp"

#define NARGS 2

using namespace std;
using namespace libconfig;



// Print usage of the program
void PrintInfo(void);

int main(int argc, char * argv[]) {
    
    // Get as first parameter the input file
    if (argc != NARGS) {
        PrintInfo();
        cerr << "Error, you must provide one argument." << endl;
        cerr << "       that specifies the input file." << endl;
        exit(EXIT_FAILURE);
    }


    ParseInput(argv[1]);

    return EXIT_SUCCESS;
}


void PrintInfo(void) {
    cout << "TEST THE ATOMIC NETWORK" << endl;
    cout << "=======================" << endl << endl;
    cout << "Usage:" << endl;
    cout << "  test_nn_prediction.exe <input_file.cfg>" << endl << endl;
    cout << " - input_file.cfg   :  The input file that specify what to do." << endl;
    cout << endl;
}