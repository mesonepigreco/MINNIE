#!/bin/bash

BIN_DIR=../../../bin
EXECUTABLE=test_nn_prediction.exe

BIN=$BIN_DIR/$EXECUTABLE

# Lets build the neural network
#$BIN generate_twosym.cfg

# Analyze the results
$BIN test_new.cfg > results.dat

# Plot the energy and the force
echo "Results saved in 'results.dat'"
echo "Plotting the data using python..."
echo "(You need numpy as matplotlib installed to see the results)"
cat > plot.py <<EOF
from numpy import *
from matplotlib.pyplot import *


data = loadtxt("results.dat")

figure()
title("Energy")
xlabel(r"X coord [$\AA$]")
ylabel(r"Energy [eV]")
plot(data[:,0], data[:,1])
tight_layout()

# Get the numeric derivative of the energy
# And compare it with the force computed by the program
dx = mean(diff(data[:,0]))
numeric_f = - gradient(data[:, 1]) / dx

figure()
title("Force test")
xlabel(r"X coord [$\AA$]")
ylabel(r"force [eV/$\AA$]")

plot(data[:,0], data[:, 2], label = "analitic force")
plot(data[:,0], numeric_f, label = "numeric force")
legend()
tight_layout()
show()
EOF
python plot.py

