# MINNIE
Minimal ab Initio Neural Network Interatomic Engine

MINNIE is a C++/Python software and library that can be used to train an atomic Neural Network 
to reproduce ab-inito total energy and force calculations.

The focus of this library is the easy-to-use and extendibility through python scripting. 
The aim of the project is to make it interfaced with tools like ASE (Atomic Simulation Environment),
to provide a fast and reliable calculator for energy, forces and stress tensors of condensed matter system as well as big organic molecules 

MINNIE is a free software, and comes as it is, without any warranty.

THIS IS AN EXPERIMENTAL VERSION, DO NOT USE IT IF YOU DO NOT KNOW WHAT YOU ARE DOING



## Requirements

The MINNIE software requires python (>= 3.6), numpy and C/C++ compilers.
To compile the C/C++ code you need:
1. libconfig. On ubuntu, it is installed with `sudo apt install libconfig++-dev`.
2. Gnu Scientific Library (GSL), with developer headers
3. Python.h headers (`sudo apt install python-dev`)


On ubuntu, the ingredients to compile the code can be installed with

```
sudo apt install libboost-all-dev libconfig++-dev libgsl-dev
```

