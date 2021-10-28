# List of the TODO to speedup

The calculation of the forces is slow.
The slowest part is the derivatives of the symmetric functions with respect to the atmonic position.
Which is taking the full timing. 
To speedupt things:

1. Return from the GetDerivatives only the interacting atoms with the ith atom.
2. When generating the supercell, exclude automatically all atoms which are more distant than the cutoff with the original atoms
