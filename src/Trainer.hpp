/*
 * This module contains the class that stores all the info to train
 * a neural network. It may also read the info from the CFG files.
 */

#ifndef H_TRAINER
#define H_TRAINER

#include <iostream>
#include <string>
#include "ensemble.hpp"
#include "AtomicNetwork.hpp"

// Setup the variables for the CFG file
#define TRAINER_KEYWORD "Trainer"
#define TRAINER_METHOD "method"
#define TRAINER_STEP "step"
#define TRAINER_NITERS "max_iterations"
#define TRAINER_USE_LMIN "line_minimization"


class Trainer {
public:
    Ensemble * training_set; 
    string method;
    double step; 
    int N_steps;
    bool use_lmin;


    // Constructor
    Trainer();

    // Constructor with initialization from the CFG file
    Trainer(const char * cfg_filename);

    // Distructor
    ~Trainer();

    // Setup all the variables from a CFG file.
    void SetupFromCFG(const char * file);

    // Train the provided network using the parameters of this class
    /* 
     * The optional second argument, the precondition, 
     * should be setted to true if it is the first time
     * the network is trained. This will scan the ensemble and 
     * setup the tailoring bias and sinapsis so that 
     * the network will reproduce the correct average energy and fluctuation.
     * 
     * It is good to set up a reasonable starting point for the network.
     */
    void TrainAtomicNetwork(AtomicNetwork * target, bool precondition = false);
};
#endif