#ifdef _PYTHON2
#include <python2.7/Python.h>
#else
#include <Python.h>
#endif
#include <numpy/arrayobject.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "ensemble.hpp"
#include "AtomicNetwork.hpp"
#include "symmetric_functions.hpp"

#include <string>

#ifdef _MPI
#include<mpi.h>
#endif

using namespace std;


// Function Prototypes
static PyObject * load_ensemble_from_dir(PyObject*self, PyObject *args);
static PyObject * print_configuration(PyObject * self, PyObject * args);
static PyObject * load_ensemble_from_cfg(PyObject*self, PyObject * args);
static PyObject * add_g2_function(PyObject * self, PyObject * args);
static PyObject * add_g4_function(PyObject * self, PyObject * args);
//static PyObject * load_symmetric_functions_from_cfg(PyObject * self, PyObject * args);
PyObject * construct_symmetry(PyObject* self, PyObject * args);
PyObject * symmetry_load_from_cfg(PyObject * self, PyObject * args);
PyObject * symmetry_save_to_cfg(PyObject * self, PyObject * args);
PyObject * atomic_network_load_from_cfg(PyObject * self, PyObject * args);
PyObject * atomic_network_save_to_cfg(PyObject * self, PyObject * args);
PyObject * construct_atoms(PyObject*self, PyObject * args);
PyObject * load_atoms(PyObject*self, PyObject * args);
PyObject * print_atoms(PyObject*self, PyObject * args);
PyObject * construct_ensemble(PyObject* self, PyObject * args);
PyObject * set_atoms_coords_type(PyObject * self, PyObject * args);
PyObject * get_atoms_coords_type(PyObject * self, PyObject * args);
PyObject * get_symmetric_functions_from_atoms(PyObject * self, PyObject * args);
PyObject * get_symmetric_functions_from_atoms_index(PyObject * self, PyObject * args);
PyObject * get_symmetric_functions_parameters(PyObject * self, PyObject * args);
PyObject * set_symmetric_functions_parameters(PyObject * self, PyObject * args);
PyObject * set_cutoff(PyObject * self, PyObject * args);
PyObject * get_cutoff(PyObject * self, PyObject * args);
PyObject * set_cutoff_type(PyObject * self, PyObject * args);
PyObject * get_n_sym_functions(PyObject * self, PyObject * args);
PyObject * sym_print_info(PyObject * self, PyObject * args);
PyObject * override_ensemble(PyObject * self, PyObject * args);
PyObject * create_atomic_network(PyObject * self, PyObject * args);
PyObject * get_covariance_matrix(PyObject * self, PyObject * args);
PyObject * get_ensemble_nconfigs(PyObject * self, PyObject * args);
PyObject * get_ensemble_config(PyObject * self, PyObject * args);
PyObject * get_ensemble_ntyps(PyObject * self, PyObject * args);
PyObject * get_n_atoms(PyObject * self, PyObject * args);
PyObject * nn_get_energy(PyObject * self, PyObject * args);
PyObject * nn_get_loss(PyObject * self, PyObject * args);
PyObject * nn_get_ntypes(PyObject * self, PyObject * args);
PyObject * nn_get_nbiases_nsynapsis(PyObject * self, PyObject * args);
PyObject * nn_get_biases_synapsis(PyObject * self, PyObject * args);
PyObject * nn_set_biases_synapsis(PyObject * self, PyObject * args);
PyObject * ensemble_get_energy_force(PyObject * self, PyObject * args);
PyObject * ensemble_shuffle(PyObject * self, PyObject * args);
PyObject * mpi_init(PyObject * self, PyObject * args);
PyObject * mpi_get_rank_size(PyObject * self, PyObject * args);

// Define the name for the capsules
#define NAME_SYMFUNC "symmetry_functions"
#define NAME_ANN "atomic_neural_networks"
#define NAME_ATOMS "atoms"
#define NAME_ENSEMBLE "ensemble"

static PyMethodDef Methods[] = {
    {"LoadEnsemble", load_ensemble_from_dir, METH_VARARGS, "Load the ensemble from a directory"},
    {"PrintConfig", print_configuration, METH_VARARGS, "Print the given configuration on stdout"},
    {"LoadEnsembleFromCFG", load_ensemble_from_cfg, METH_VARARGS, "Load the ensemble from a configuration file."},
    {"AddSymG2", add_g2_function, METH_VARARGS, "Add a G2 symmetry function"},
    {"AddSymG4", add_g4_function, METH_VARARGS, "Add a G4 symmetry function"},
    {"CreateSymFuncClass", construct_symmetry, METH_VARARGS, "Create the symmetric function class"},
    {"LoadSymFuncFromCFG", symmetry_load_from_cfg, METH_VARARGS, "Load the symmetric function from a configuration file"},
    {"SaveSymFuncToCFG", symmetry_save_to_cfg, METH_VARARGS, "Save the symmetric function to a configuration file"},
    {"CreateAtomsClass", construct_atoms, METH_VARARGS, "Create the Atoms class"},
    {"LoadSCF", load_atoms, METH_VARARGS, "Load an SCF file to create an Atoms class"},
    {"PrintAtoms", print_atoms, METH_VARARGS, "Print on STDOUT the current atoms"},
    {"GetNAtoms", get_n_atoms, METH_VARARGS, "Get how many atoms are inside an Atom class"},
    {"GetEnsembleConfig", get_ensemble_config, METH_VARARGS, "Get the ensemble configuration"},
    {"SetAtomsCoordsTypes", set_atoms_coords_type, METH_VARARGS, "Set from python the Atoms class attributes"},
    {"GetAtomsCoordsTypes", get_atoms_coords_type, METH_VARARGS, "Get the coords and types of the atom"},
    {"GetSymmetricFunctions", get_symmetric_functions_from_atoms, METH_VARARGS, "Get the symmetric functions for the atoms class"},
    {"GetSymmetricFunctionsAtomIndex", get_symmetric_functions_from_atoms_index, METH_VARARGS, "Get the symmetric functions for the atoms class"},
    {"GetCovarianceMatrix", get_covariance_matrix, METH_VARARGS, "Get the covariance matrix of symmetric functions on ensemble"},
    {"GetSymmetricFunctionParameters", get_symmetric_functions_parameters, METH_VARARGS, "Get the parameters of the symmetric function."},
    {"SetSymmetricFunctionParameters", set_symmetric_functions_parameters, METH_VARARGS, "Set the parameters of the symmetric function."},
    {"SetCutoffRadius", set_cutoff, METH_VARARGS, "Set the cutoff radius."},
    {"SetCutoffType", set_cutoff_type, METH_VARARGS, "Set the cutoff funciton type."},
    {"GetCutoffTypeRadius", get_cutoff, METH_VARARGS, "Get the cutoff funciton (type and radius)."},
    {"SymPrintInfo", sym_print_info, METH_VARARGS, "Print Info about the symmetric functions"},
    {"GetNSyms", get_n_sym_functions, METH_VARARGS, "Get the number of symmetric functions."},
    {"LoadNNFromCFG", atomic_network_load_from_cfg, METH_VARARGS, "Load the NN from the configuration file"},
    {"SaveNNToCFG", atomic_network_save_to_cfg, METH_VARARGS, "Save the NN into the configuration file"},
    {"GetNConfigsEnsemble", get_ensemble_nconfigs, METH_VARARGS, "Get the number of configurations in an ensemble"},
    {"GetNTypsEnsemble", get_ensemble_ntyps, METH_VARARGS, "Get the number of atomic species in the ensemble"},
    {"CreateEnsembleClass", construct_ensemble, METH_VARARGS, "Create an empty ensemble."},
    {"OvverrideEnsembleIndex", override_ensemble, METH_VARARGS, "Override the i-th structure of the ensemble."},
    {"CreateAtomicNN", create_atomic_network, METH_VARARGS, "Create a new Atomic NN from ensemble"},
    {"NN_GetEnergy", nn_get_energy, METH_VARARGS, "Get energies and forces from an Atomic NN."},
    {"NN_GetLoss", nn_get_loss, METH_VARARGS, "Get the loss function of the ANN"},
    {"NN_GetNTypes", nn_get_ntypes, METH_VARARGS, "Get the number of types"},
    {"NN_GetNBiasesSynapsis", nn_get_nbiases_nsynapsis, METH_VARARGS, "Get the number of biases and synaptics in a network"},
    {"NN_GetBiasesSynapsis", nn_get_biases_synapsis, METH_VARARGS, "Get the biases and synaptics in all the atomic networks"},
    {"NN_SetBiasesSynapsis", nn_set_biases_synapsis, METH_VARARGS, "Set the biases and synaptics in all the atomic networks"},
    {"Ensemble_GetEnergyForces", ensemble_get_energy_force, METH_VARARGS, "Get the energy and forces for a configuration of the ensemble"},
    {"Ensemble_Shuffle", ensemble_shuffle, METH_VARARGS, "Get the energy and forces for a configuration of the ensemble"},
    {"MPI_init", mpi_init, METH_VARARGS, "Initialize the parallel environment"},
    {"MPI_get_rank_size", mpi_get_rank_size, METH_VARARGS, "Get the MPI rank and size"},
    {NULL, NULL, 0, NULL}
};

// Module initialization
#ifdef _PYTHON2
PyMODINIT_FUNC initNNcpp(void) {
    (void) Py_InitModule("NNcpp", Methods);
}
#else
static struct PyModuleDef NNcpp = {
  PyModuleDef_HEAD_INIT, 
  "NNcpp", 
  NULL, 
  -1, 
  Methods
};

PyMODINIT_FUNC PyInit_NNcpp(void) {
  return PyModule_Create(&NNcpp);
}
#endif

// ---------------------------------- FROM NOW ON THE CODE ---------------------------------

// Define the ensemble as a general variable in memory
//Ensemble * ensemble = NULL;
//SymmetricFunctions * sym_functs = NULL;

static PyObject * load_ensemble_from_dir(PyObject * self, PyObject * args) {
    const char * path_dir;
    int N_configs, pop, N_atoms;
    PyObject * py_ens;
    double alat;
    int ovrd;
    bool overwrite = false;

    // Get the path dir
    if (!PyArg_ParseTuple(args, "Osiiidp", &py_ens, &path_dir, &N_configs, &pop, &N_atoms, &alat, &ovrd))
        return NULL;
    
    if (ovrd)
        overwrite = true;

    Ensemble * ensemble = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);

    // Load the ensemble
    string path(path_dir);
    ensemble->Load(path, N_configs, pop, N_atoms, alat, overwrite);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * print_configuration(PyObject * self, PyObject * args) {
    int index;

    if (! PyArg_ParseTuple(args, "i", &index)) {
        return NULL;
    }
/* 
    if (!ensemble) {
        cerr << "Error, you must allocate the ensemble before." << endl;
        return NULL;
    }

    // Print on stdout the coordinates
    Atoms * conf;
    ensemble->GetConfig(index, conf);

    conf->PrintCoords(); */

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * load_ensemble_from_cfg(PyObject*self, PyObject * args) {
    const char * cfg_file;
    PyObject * py_ens;
    if(!PyArg_ParseTuple(args, "Os", &py_ens, &cfg_file))  {
        cerr << "Error, we require 2 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " line " << __LINE__ << endl;
        return NULL;

    }
    
    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);

    // Load the ensemble
    ens->LoadFromCFG(cfg_file);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * add_g2_function(PyObject * self, PyObject * args) {
    double rs, eta;
    PyObject * symFuncs;
    if (!PyArg_ParseTuple(args, "Odd", &symFuncs, &rs, &eta)) {
        cerr << "Error, this function requires The sym function, rs and eta (double type)" << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    SymmetricFunctions* sym_funcs = (SymmetricFunctions*) PyCapsule_GetPointer(symFuncs, NAME_SYMFUNC);
    sym_funcs->AddG2Function(rs, eta);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * add_g4_function(PyObject * self, PyObject * args) {
    double eta, zeta;
    int lambda;
    PyObject * symFuncs;
    if (!PyArg_ParseTuple(args, "Oddi", &symFuncs, &eta, &zeta, &lambda)) {
        cerr << "Error, this function requires sym fuunctions eta, zeta(double type) and lambda (int)" << endl;
        return NULL;
    }

    SymmetricFunctions* sym_funcs = (SymmetricFunctions*) PyCapsule_GetPointer(symFuncs, NAME_SYMFUNC);


    sym_funcs->AddG4Function(eta, zeta, lambda);

    Py_INCREF(Py_None);
    return Py_None;
}
/* 
static PyObject * load_symmetric_functions_from_cfg(PyObject * self, PyObject * args) {
    const char * config_file;
    if (!PyArg_ParseTuple(args, "s", &config_file)) {
        cerr << "Error, 1 argument (string) required" << endl;
        return NULL;
    }

    // Reset the symmetric functions
    if (!sym_functs) {
        sym_functs = new SymmetricFunctions();
    } else {
        delete sym_functs;
        sym_functs = new SymmetricFunctions();
    }

    sym_functs->LoadFromCFG(config_file);

    Py_INCREF(Py_None);
    return Py_None;
} */

/*
 * Prepare a python constructor for the symmetry file function
 */
PyObject * construct_symmetry(PyObject* self, PyObject * args) {
    //const char * object_name = "nonamed";

    //PyArg_ParseTuple(args, "s", &object_name);

    // Allocate the memory for the Symmetry Function Class
    SymmetricFunctions* sym_funcs = new SymmetricFunctions();

    // Prepare the python object for the symmetric function
    PyObject* sym_funcs_capsule = PyCapsule_New( (void*) sym_funcs, NAME_SYMFUNC, NULL);
    PyCapsule_SetPointer(sym_funcs_capsule, (void*) sym_funcs);

    // Return to python the sym func capsule
    return Py_BuildValue("O", sym_funcs_capsule);
}

// The cosnstructor for the atoms
PyObject * construct_atoms(PyObject* self, PyObject * args) {
    // Allocate the memory for the Symmetry Function Class
    int Natoms;
    if (!PyArg_ParseTuple(args, "i", &Natoms)) {
        cerr << "Error, I need to know the exact number of atoms before the allocation." << endl;
        return NULL;
    }

    Atoms* atoms = new Atoms(Natoms);

    // Prepare the python object for the symmetric function
    PyObject* atoms_capsule = PyCapsule_New( (void*) atoms, NAME_ATOMS, NULL);
    PyCapsule_SetPointer(atoms_capsule, (void*) atoms);

    // Return to python the sym func capsule
    return Py_BuildValue("O", atoms_capsule);
}


// The cosnstructor for the atoms
PyObject * get_ensemble_config(PyObject* self, PyObject * args) {
    // Allocate the memory for the Symmetry Function Class
    PyObject * py_ens;
    int index;
    if (!PyArg_ParseTuple(args, "Oi", &py_ens, &index)) {
        cerr << "Error, I need to know the exact number of atoms before the allocation." << endl;
        return NULL;
    }

    Atoms* atoms;
    Ensemble* ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);
    ens->GetConfig(index, atoms);

    // Prepare the python object for the symmetric function
    PyObject* atoms_capsule = PyCapsule_New( (void*) atoms, NAME_ATOMS, NULL);
    PyCapsule_SetPointer(atoms_capsule, (void*) atoms);

    // Return to python the sym func capsule
    return Py_BuildValue("O", atoms_capsule);
}
/*
 * Prepare a python constructor for the symmetry file function
 */
PyObject * construct_ensemble(PyObject* self, PyObject * args) {
    //const char * object_name = "nonamed";

    //PyArg_ParseTuple(args, "s", &object_name);
    // Allocate the memory for the Symmetry Function Class
    int Natoms, Nconfigs;
    if (!PyArg_ParseTuple(args, "ii", &Nconfigs, &Natoms)) {
        cerr << "Error, I need to know the exact number of structures and atoms per structure before the allocation." << endl;
        return NULL;
    }
    // Allocate the memory for the Symmetry Function Class
    Ensemble* ens = new Ensemble(Nconfigs, Natoms);

    cout << "The builded ensemble has " << ens->GetNConfigs() << endl;

    // Prepare the python object for the symmetric function
    PyObject* ens_cap = PyCapsule_New( (void*) ens, NAME_ENSEMBLE, NULL);
    PyCapsule_SetPointer(ens_cap, (void*) ens);

    // Return to python the sym func capsule
    return Py_BuildValue("O", ens_cap);
}


PyObject * symmetry_load_from_cfg(PyObject* self, PyObject * args) {
    const char * fname;
    PyObject * symFuncs;


    if (!PyArg_ParseTuple(args, "Os", &symFuncs, &fname)) {
        cerr << "Error this function requires 2 arguments:" << endl;
        cerr << "The symmetry function class and the file name." << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    SymmetricFunctions* sym_funcs = (SymmetricFunctions*) PyCapsule_GetPointer(symFuncs, NAME_SYMFUNC);

    // Load the symmetric function from the given file
    sym_funcs->LoadFromCFG(fname);

    // Return none
    return Py_BuildValue("");
}


PyObject * load_atoms(PyObject* self, PyObject * args) {
    const char * fname;
    PyObject * py_atoms;


    if (!PyArg_ParseTuple(args, "Os", &py_atoms, &fname)) {
        cerr << "Error this function requires 2 arguments:" << endl;
        cerr << "The symmetry function class and the file name." << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    Atoms*  atoms= (Atoms*) PyCapsule_GetPointer(py_atoms, NAME_ATOMS);

    // Load the symmetric function from the given file
    atoms->ReadSCF(fname);

    // Return none
    return Py_BuildValue("");
}


PyObject * print_atoms(PyObject* self, PyObject * args) {
    PyObject * py_atoms;


    if (!PyArg_ParseTuple(args, "O", &py_atoms)) {
        cerr << "Error this function requires 1 arguments:" << endl;
        cerr << "The symmetry function class and the file name." << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    Atoms*  atoms= (Atoms*) PyCapsule_GetPointer(py_atoms, NAME_ATOMS);

    // Load the symmetric function from the given file
    atoms->PrintCoords();

    // Return none
    return Py_BuildValue("");
}


/*
 * Save the symmetric function on a file
 */
PyObject * symmetry_save_to_cfg(PyObject* self, PyObject * args) {
    const char * fname;
    PyObject * symFuncs;


    if (!PyArg_ParseTuple(args, "Os", &symFuncs, &fname)) {
        cerr << "Error this function requires 2 arguments:" << endl;
        cerr << "The symmetry function class and the file name." << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    SymmetricFunctions* sym_funcs = (SymmetricFunctions*) PyCapsule_GetPointer(symFuncs, NAME_SYMFUNC);

    // Load the symmetric function from the given file
    sym_funcs->SaveToCFG(fname);

    // Return none
    return Py_BuildValue("");
}


PyObject * set_atoms_coords_type(PyObject * self, PyObject * args) {
    PyArrayObject * npy_coords, *npy_types, *npy_uc;
    PyObject * atoms;
    double * coords, *uc;
    int *types;
    int N_atoms;

    if (!PyArg_ParseTuple(args, "OOOO", &atoms, &npy_coords, &npy_types, &npy_uc)) {
        cerr << "Error, this function requires 3 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the atoms
    Atoms * atm = (Atoms*) PyCapsule_GetPointer(atoms, NAME_ATOMS);
    N_atoms = atm->GetNAtoms();

    // Get the raw data from the numpy arrays
    // NOTE: Here something can go wrong
    //       please, always be carefull when passing data type to native C functions
    coords = (double*) PyArray_DATA(npy_coords);
    types = (int*) PyArray_DATA(npy_types);
    uc = (double * ) PyArray_DATA(npy_uc);

    // Fill the atoms with the correct data
    for (int i = 0; i < N_atoms; ++i) {
        atm->types[i] = types[i];
        for (int j = 0; j <3; ++j) {
            atm->coords[3 * i + j] = coords[3*i+j];
        }
    }
    for (int i = 0; i < 3; ++i) 
        for (int j = 0; j < 3; ++j) 
            atm->unit_cell[3*i +j] = uc[3*i +j];

    return Py_BuildValue("");
}



PyObject * get_atoms_coords_type(PyObject * self, PyObject * args) {
    PyArrayObject * npy_coords, *npy_types, *npy_uc;;
    PyObject * atoms;
    double * coords, *uc;
    int *types;
    int N_atoms;

    if (!PyArg_ParseTuple(args, "OOOO", &atoms, &npy_coords, &npy_types, &npy_uc)) {
        cerr << "Error, this function requires 3 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the atoms
    Atoms * atm = (Atoms*) PyCapsule_GetPointer(atoms, NAME_ATOMS);
    N_atoms = atm->GetNAtoms();

    // Get the raw data from the numpy arrays
    // NOTE: Here something can go wrong
    //       please, always be carefull when passing data type to native C functions
    coords = (double*) PyArray_DATA(npy_coords);
    uc = (double*) PyArray_DATA(npy_uc);
    types = (int*) PyArray_DATA(npy_types);

    // Fill the atoms with the correct data
    for (int i = 0; i < N_atoms; ++i) {
        types[i] = atm->types[i];
        for (int j = 0; j <3; ++j) {
            coords[3 * i + j] = atm->coords[3*i+j];
        }
    }
    for (int i = 0; i < 3; ++i) 
        for (int j = 0; j < 3; ++j) 
            uc[3*i +j] = atm->unit_cell[3*i +j];

    return Py_BuildValue("");
}


PyObject * get_n_atoms(PyObject * self, PyObject * args) {
    PyObject * atoms;
    int N_atoms;

    if (!PyArg_ParseTuple(args, "O", &atoms)) {
        cerr << "Error, this function requires 1arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the atoms
    Atoms * atm = (Atoms*) PyCapsule_GetPointer(atoms, NAME_ATOMS);
    N_atoms = atm->GetNAtoms();

    return Py_BuildValue("i", N_atoms);
}

PyObject* get_symmetric_functions_from_atoms(PyObject * self, PyObject * args) {
    PyObject * symf, *atm;
    PyArrayObject * py_output;
    int N_atoms;
    int N_syms, n_types;
    int Nx, Ny, Nz; // The periodic images of the atoms

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "OOiiiiO", &symf, &atm, &n_types, &Nx, &Ny, &Nz, &py_output)) {
        cerr << "Error, this function requires 5 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);
    Atoms * atoms = (Atoms*) PyCapsule_GetPointer(atm, NAME_ATOMS);
    N_atoms = atoms->GetNAtoms();
    N_syms = symm_func->GetTotalNSym(n_types);

    // Allocate the symmfunction arrays
    double * sym_coords = (double*) PyArray_DATA(py_output);

    // Calculate the symmetric coordinates
    symm_func->GetSymmetricFunctions(atoms, Nx, Ny, Nz, sym_coords, n_types);

    return Py_BuildValue("");
}

PyObject* get_symmetric_functions_from_atoms_index(PyObject * self, PyObject * args) {
    PyObject * symf, *atm;
    PyArrayObject * py_output;
    int N_atoms;
    int index;
    int N_syms, n_types;
    int Nx, Ny, Nz; // The periodic images of the atoms

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "OOiiiiiO", &symf, &atm, &index, &n_types, &Nx, &Ny, &Nz, &py_output)) {
        cerr << "Error, this function requires 5 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);
    Atoms * atoms = (Atoms*) PyCapsule_GetPointer(atm, NAME_ATOMS);
    N_atoms = atoms->GetNAtoms();
    N_syms = symm_func->GetTotalNSym(n_types);

    // Allocate the symmfunction arrays
    double * sym_coords = (double*) PyArray_DATA(py_output);

    // Calculate the symmetric coordinates
    symm_func->GetSymmetricFunctionsATM(atoms, Nx, Ny, Nz, sym_coords, index, n_types);

    return Py_BuildValue("");
}



PyObject* sym_print_info(PyObject * self, PyObject * args) {
    PyObject * symf;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "O", &symf)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    symm_func->PrintInfo();

    return Py_BuildValue("");
}
PyObject* get_symmetric_functions_parameters(PyObject * self, PyObject * args) {
    PyObject * symf;
    int index, g2or4;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "Oii", &symf, &index, &g2or4)) {
        cerr << "Error, this function requires 3 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    double p1, p2;
    int p3;
    if (g2or4) {
        symm_func->GetG2Parameters(index, p1, p2);
        return Py_BuildValue("dd", p1, p2);
    } else {
        symm_func->GetG4Parameters(index, p1, p2, p3);
        return Py_BuildValue("ddi", p1, p2, p3);
    }
}
PyObject* set_symmetric_functions_parameters(PyObject * self, PyObject * args) {
    PyObject * symf;
    int index, g2or4;

    double p1, p2;
    int p3;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "Oiiddi", &symf, &index, &g2or4, &p1, &p2, &p3)) {
        cerr << "Error, this function requires 6 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    if (g2or4) {
        symm_func->EditG2Function(index, p1, p2);
        return Py_BuildValue("");
    } else {
        symm_func->EditG4Function(index, p1, p2, p3);
        return Py_BuildValue("");
    }
}

PyObject* set_cutoff(PyObject * self, PyObject * args) {
    PyObject * symf;

    double cutoff;
    int type;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "Od", &symf, &cutoff)) {
        cerr << "Error, this function requires 2 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    type = symm_func->get_cutoff_type();
    //cout << "type = " << type << endl;
    symm_func->SetupCutoffFunction(type, cutoff);
    return Py_BuildValue("");
}

PyObject* set_cutoff_type(PyObject * self, PyObject * args) {
    PyObject * symf;

    double cutoff;
    int type;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "Oi", &symf, &type)) {
        cerr << "Error, this function requires 2 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    cutoff = symm_func->get_cutoff();
    //cout << "type = " << type << endl;
    symm_func->SetupCutoffFunction(type, cutoff);
    return Py_BuildValue("");
}


PyObject* get_cutoff(PyObject * self, PyObject * args) {
    PyObject * symf;

    double cutoff;
    int type;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "O", &symf)) {
        cerr << "Error, this function requires 2 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    cutoff = symm_func->get_cutoff();
    type = symm_func->get_cutoff_type();
    return Py_BuildValue("id", type, cutoff);
}



PyObject* get_n_sym_functions(PyObject * self, PyObject * args) {
    PyObject * symf;
    int ntyps;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "Oi", &symf, &ntyps)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    int n2, n4;
    n2 = symm_func->get_n_g2();
    n4 = symm_func->get_n_g4();

    return Py_BuildValue("iii", n2, n4, symm_func->GetTotalNSym(ntyps));
}


PyObject * atomic_network_load_from_cfg(PyObject* self, PyObject * args) {
    const char * fname;


    if (!PyArg_ParseTuple(args, "s", &fname)) {
        cerr << "Error this function requires 1 arguments:" << endl;
        cerr << "the file name." << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Allocate the memory for the Symmetry Function Class
    AtomicNetwork* ann = new AtomicNetwork(fname);

    // Prepare the python object for the symmetric function
    PyObject* ann_cap = PyCapsule_New( (void*) ann, NAME_ANN, NULL);
    PyCapsule_SetPointer(ann_cap, (void*) ann);

    // Return to python the sym func capsule
    return Py_BuildValue("O", ann_cap);
}

PyObject * create_atomic_network(PyObject* self, PyObject * args) {

    PyObject * py_ens, *py_symf;
    PyArrayObject * py_hidden;
    int * hidden_layer;
    int n_hidden, n_lim;


    if (!PyArg_ParseTuple(args, "OOiiO", &py_symf, &py_ens, &n_lim, &n_hidden, &py_hidden)) {
        cerr << "Error this function requires 5 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(py_symf, NAME_SYMFUNC);
    Ensemble* ensemble = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);

    //       please, always be carefull when passing data type to native C functions
    hidden_layer = (int*) PyArray_DATA(py_hidden);

    // Allocate the memory for the Symmetry Function Class
    AtomicNetwork* ann = new AtomicNetwork(symm_func, ensemble, n_lim, n_hidden, hidden_layer, 0);

    // Prepare the python object for the symmetric function
    PyObject* ann_cap = PyCapsule_New( (void*) ann, NAME_ANN, NULL);
    PyCapsule_SetPointer(ann_cap, (void*) ann);

    // Return to python the sym func capsule
    return Py_BuildValue("O", ann_cap);
}

/*
 * Save the symmetric function on a file
 */
PyObject * atomic_network_save_to_cfg(PyObject* self, PyObject * args) {
    const char * fname;
    PyObject * ann;


    if (!PyArg_ParseTuple(args, "Os", &ann, &fname)) {
        cerr << "Error this function requires 2 arguments:" << endl;
        cerr << "The symmetry function class and the file name." << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    AtomicNetwork* myann = (AtomicNetwork*) PyCapsule_GetPointer(ann, NAME_ANN);

    // Load the symmetric function from the given file
    myann->SaveCFG(fname);

    // Return none
    return Py_BuildValue("");
}


/*
 * Override the ensemble
 */
PyObject * override_ensemble(PyObject* self, PyObject * args) {
    PyObject * py_ens;
    PyObject * py_atm;

    PyArrayObject * npy_forces, *npy_stresses;
    double * forces, *stresses;
    double energy;
    int index;

    if (!PyArg_ParseTuple(args, "iOdOOO", &index, &py_ens, &py_atm, &energy, &npy_forces, &npy_stresses)) {
        cerr << "Error, this function requires 5 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the atoms
    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);
    Atoms * atm = (Atoms*) PyCapsule_GetPointer(py_atm, NAME_ATOMS);


    // Get the raw data from the numpy arrays
    // NOTE: Here something can go wrong
    //       please, always be carefull when passing data type to native C functions
    forces = (double*) PyArray_DATA(npy_forces);
    stresses = (double*) PyArray_DATA(npy_stresses);

    ens->SetConfig(index, atm, energy, forces, stresses);


    // Return none
    return Py_BuildValue("");
}


PyObject * get_covariance_matrix(PyObject * self, PyObject * args) {
    PyObject * py_ens;
    PyObject * py_sym;
    PyArrayObject * py_means, *py_cvar_mat;
    int Nx, Ny, Nz;
    

    if (!PyArg_ParseTuple(args, "OOiiiOO", &py_ens, &py_sym, &Nx, &Ny, &Nz, &py_means, &py_cvar_mat)) {
        cerr << "Error, this function requires 7 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);
    SymmetricFunctions * symf = (SymmetricFunctions*) PyCapsule_GetPointer(py_sym, NAME_SYMFUNC);

    int n_typ = ens->GetNTyp();
    int n_sym = symf->GetTotalNSym(n_typ);

    double * means = (double*) PyArray_DATA(py_means);
    double * cvar_mat = (double*) PyArray_DATA(py_cvar_mat);


    GetCovarianceSymmetry(ens, symf, Nx, Ny, Nz, means, cvar_mat);

    return Py_BuildValue("");
}
PyObject * get_ensemble_nconfigs(PyObject* self, PyObject * args) {
    PyObject * py_ens;
    int ncfgs;


    if (!PyArg_ParseTuple(args, "O", &py_ens)) {
        cerr << "Error this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);
    ncfgs = ens->GetNConfigs();

    return Py_BuildValue("i", ncfgs);
}


PyObject * get_ensemble_ntyps(PyObject* self, PyObject * args) {
    PyObject * py_ens;
    int ntyp;


    if (!PyArg_ParseTuple(args, "O", &py_ens)) {
        cerr << "Error this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);
    ntyp = ens->GetNTyp();

    return Py_BuildValue("i", ntyp);
}



PyObject * nn_get_energy(PyObject * self, PyObject * args) {
    PyObject * py_ann;
    PyObject * py_atoms;
    PyArrayObject * py_forces;
    int Nx, Ny, Nz;
    int get_forces;
    double energy;
    

    if (!PyArg_ParseTuple(args, "OOpOiii", &py_ann, &py_atoms, &get_forces, &py_forces, &Nx, &Ny, &Nz)) {
        cerr << "Error, this function requires 7 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    AtomicNetwork * ann = (AtomicNetwork*) PyCapsule_GetPointer(py_ann, NAME_ANN);
    Atoms * atoms = (Atoms*) PyCapsule_GetPointer(py_atoms, NAME_ATOMS);

    double * forces = NULL;
    if (get_forces)
        forces = (double*) PyArray_DATA(py_forces);


    energy = ann->GetEnergy(atoms, forces, Nx, Ny, Nz);

    return Py_BuildValue("d", energy);
}



PyObject * nn_get_loss(PyObject * self, PyObject * args) {
    PyObject * py_ann;
    PyObject * py_ensemble;
    PyArrayObject * py_biases, *py_synapsis;
    int offset, ncfg;
    double energy_weight, force_weight;
    

    if (!PyArg_ParseTuple(args, "OOddOOii", &py_ann, &py_ensemble, &energy_weight, &force_weight, &py_biases, &py_synapsis, &offset, &ncfg)) {
        cerr << "Error, this function requires 7 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    AtomicNetwork * ann = (AtomicNetwork*) PyCapsule_GetPointer(py_ann, NAME_ANN);
    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ensemble, NAME_ENSEMBLE);

    double * grad_biases =  (double*) PyArray_DATA(py_biases);
    double * grad_synapsis = (double*) PyArray_DATA(py_synapsis);

    double ** all_grads_b = (double**) malloc(sizeof(double*) * ann->N_types);
    double ** all_grads_s = (double**) malloc(sizeof(double*) * ann->N_types);
    for (int i = 0; i < ann->N_types; ++i) {
        all_grads_b[i] = grad_biases   + i * ann->GetNNFromElement(i)->get_nbiases();
        all_grads_s[i] = grad_synapsis + i * ann->GetNNFromElement(i)->get_nsinapsis();
    }


    double loss = ann->GetLossGradient(ens, energy_weight, force_weight, all_grads_b, all_grads_s, offset, ncfg);

    free(all_grads_b);
    free(all_grads_s);

    return Py_BuildValue("d", loss);
}



PyObject * nn_get_ntypes(PyObject * self, PyObject * args) {
    PyObject * py_ann;
    int ntyps;
    

    if (!PyArg_ParseTuple(args, "O", &py_ann)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    AtomicNetwork * ann = (AtomicNetwork*) PyCapsule_GetPointer(py_ann, NAME_ANN);

    return Py_BuildValue("i", ann->N_types);
}

PyObject * nn_get_nbiases_nsynapsis(PyObject * self, PyObject * args) {
    PyObject * py_ann;
    int ntyps;
    

    if (!PyArg_ParseTuple(args, "O", &py_ann)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    AtomicNetwork * ann = (AtomicNetwork*) PyCapsule_GetPointer(py_ann, NAME_ANN);

    return Py_BuildValue("ii",  ann->GetNNFromElement(0)->get_nbiases(),  ann->GetNNFromElement(0)->get_nsinapsis());
}

PyObject * nn_get_biases_synapsis(PyObject * self, PyObject * args) {
    PyObject * py_ann;
    PyArrayObject * py_biases, *py_synapsis;
    int n_biases, n_synapsis;
    

    if (!PyArg_ParseTuple(args, "OOO", &py_ann, &py_biases, &py_synapsis)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    AtomicNetwork * ann = (AtomicNetwork*) PyCapsule_GetPointer(py_ann, NAME_ANN);
    NeuralNetwork * network;

    double * biases = (double*) PyArray_DATA(py_biases);
    double * synapsis = (double*) PyArray_DATA(py_synapsis);
    for (int i = 0; i < ann->N_types; ++i) {
        network = ann->GetNNFromElement(i);
        n_biases = network->get_nbiases();
        n_synapsis =  network->get_nsinapsis();

        for (int j = 0; j < n_biases; ++j) 
            biases[n_biases * i + j] = network->get_biases_value(j);
        for (int j = 0; j < n_synapsis; ++j)
            synapsis[n_synapsis * i + j] = network->get_sinapsis_value(j);
    }


    return Py_BuildValue("");
}

PyObject * nn_set_biases_synapsis(PyObject * self, PyObject * args) {
    PyObject * py_ann;
    PyArrayObject * py_biases, *py_synapsis;
    int n_biases, n_synapsis;
    

    if (!PyArg_ParseTuple(args, "OOO", &py_ann, &py_biases, &py_synapsis)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    AtomicNetwork * ann = (AtomicNetwork*) PyCapsule_GetPointer(py_ann, NAME_ANN);
    NeuralNetwork * network;

    double * biases = (double*) PyArray_DATA(py_biases);
    double * synapsis = (double*) PyArray_DATA(py_synapsis);
    for (int i = 0; i < ann->N_types; ++i) {
        network = ann->GetNNFromElement(i);
        n_biases = network->get_nbiases();
        n_synapsis =  network->get_nsinapsis();

        for (int j = 0; j < n_biases; ++j) 
            network->set_biases_value(j, biases[n_biases * i + j]);
        for (int j = 0; j < n_synapsis; ++j)
            network->set_sinapsis_value(j, synapsis[n_synapsis * i + j]);
    }

    return Py_BuildValue("");
}



PyObject * ensemble_get_energy_force(PyObject * self, PyObject * args) {
    PyObject * py_ens;
    PyArrayObject *py_forces;
    int index;
    int nat;
    double energy;

    if (!PyArg_ParseTuple(args, "OOii", &py_ens, &py_forces, &index, &nat)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);

    double * force = (double*) PyArray_DATA(py_forces);
    energy = ens->GetEnergy(index);

    for (int i =0; i < nat; ++ i) 
        for (int x =0; x < 3; ++x)
            force[3*i + x] = ens->GetForce(index, i, x);


    return Py_BuildValue("d", energy);
}


PyObject * ensemble_shuffle(PyObject * self, PyObject * args) {
    PyObject * py_ens;

    if (!PyArg_ParseTuple(args, "O", &py_ens)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error in file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    Ensemble * ens = (Ensemble*) PyCapsule_GetPointer(py_ens, NAME_ENSEMBLE);
    ens->Shuffle();

    return Py_BuildValue("");
}

PyObject * mpi_init(PyObject * self, PyObject * args) {

    #ifdef _MPI
    int initialized = MPI_Initialized(NULL);
    if (!initialized)
        MPI_Init(NULL, NULL);
    #endif

    return Py_BuildValue("");
}


PyObject * mpi_get_rank_size(PyObject * self, PyObject * args) {

    int rank= 0, size = 1;
    #ifdef _MPI
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    return Py_BuildValue("ii", rank, size);
}