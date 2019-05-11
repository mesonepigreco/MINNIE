#ifndef PARSE_INPUT
#define PARSE_INPUT
#include "AtomicNetwork.hpp"

/*
 * This function parses all the input file
 * and starts the appropriate calculation.
 */
void ParseInput(const char * inputfile) ;


/*
 * This function geneates the network from the input.
 */
void GetNetworkFromInput(const char * inputfile, AtomicNetwork * &network) ;
#endif