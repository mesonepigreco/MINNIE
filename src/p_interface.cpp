#include <python2.7/Python.h>
#include <numpy/arrayobject.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "ensemble.hpp"
#include "symmetric_functions.hpp"

#include <string>

using namespace std;


// Function Prototypes
static PyObject * load_ensemble_from_dir(PyObject*self, PyObject *args);
static PyObject * print_configuration(PyObject * self, PyObject * args);
static PyObject * load_ensemble_from_cfg(PyObject*self, PyObject * args);
static PyObject * add_g2_function(PyObject * self, PyObject * args);
static PyObject * add_g4_function(PyObject * self, PyObject * args);
static PyObject * load_symmetric_functions_from_cfg(PyObject * self, PyObject * args);
PyObject * construct_symmetry(PyObject* self, PyObject * args);
PyObject * symmetry_load_from_cfg(PyObject * self, PyObject * args);
PyObject * symmetry_save_to_cfg(PyObject * self, PyObject * args);
PyObject * construct_atoms(PyObject*self, PyObject * args);
PyObject * set_atoms_coords_type(PyObject * self, PyObject * args);
PyObject * get_symmetric_functions_from_atoms(PyObject * self, PyObject * args);
// Define the name for the capsules
#define NAME_SYMFUNC "symmetry_functions"
#define NAME_ATOMS "atoms"
#define NAME_ENSEMBLE "ensemble"

static PyMethodDef Methods[] = {
    {"LoadEnsemble", load_ensemble_from_dir, METH_VARARGS, "Load the ensemble from a directory"},
    {"PrintConfig", print_configuration, METH_VARARGS, "Print the given configuration on stdout"},
    {"LoadEnsembleFromCFG", load_ensemble_from_cfg, METH_VARARGS, "Load the ensemble from a configuration file."},
    {"AddSymG2", add_g2_function, METH_VARARGS, "Add a G2 symmetry function"},
    {"AddSymG4", add_g4_function, METH_VARARGS, "Add a G4 symmetry function"},
    {"LoadSymmetricFunctionsFromCFG", load_symmetric_functions_from_cfg, METH_VARARGS, "Load a set of symmetric functions from a configuration file."},
    {"CreateSymFuncClass", construct_symmetry, METH_VARARGS, "Create the symmetric function class"},
    {"LoadSymFuncFromCFG", symmetry_load_from_cfg, METH_VARARGS, "Load the symmetric function from a configuration file"},
    {"SaveSymFuncToCFG", symmetry_save_to_cfg, METH_VARARGS, "Save the symmetric function to a configuration file"},
    {"CreateAtomsClass", construct_atoms, METH_VARARGS, "Create the Atoms class"},
    {"SetAtomsCoordsTypes", set_atoms_coords_type, METH_VARARGS, "Set from python the Atoms class attributes"},
    {"GetSymmetricFunctions", get_symmetric_functions_from_atoms, METH_VARARGS, "Get the symmetric functions for the atoms class"},
    {NULL, NULL, 0, NULL}
};

// Module initialization
PyMODINIT_FUNC initNNcpp(void) {
    (void) Py_InitModule("NNcpp", Methods);
}

// ---------------------------------- FROM NOW ON THE CODE ---------------------------------

// Define the ensemble as a general variable in memory
Ensemble * ensemble = NULL;
SymmetricFunctions * sym_functs = NULL;

static PyObject * load_ensemble_from_dir(PyObject * self, PyObject * args) {
    const char * path_dir;
    int N_configs, pop, N_atoms;
    double alat;

    // Get the path dir
    if (!PyArg_ParseTuple(args, "siiid", &path_dir, &N_configs, &pop, &N_atoms, &alat))
        return NULL;

    if (!ensemble) {
        ensemble = new Ensemble();
    }

    // Load the ensemble
    string path(path_dir);
    ensemble->Load(path, N_configs, pop, N_atoms, alat);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * print_configuration(PyObject * self, PyObject * args) {
    int index;

    if (! PyArg_ParseTuple(args, "i", &index)) {
        return NULL;
    }

    if (!ensemble) {
        cerr << "Error, you must allocate the ensemble before." << endl;
        return NULL;
    }

    // Print on stdout the coordinates
    Atoms * conf;
    ensemble->GetConfig(index, conf);

    conf->PrintCoords();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * load_ensemble_from_cfg(PyObject*self, PyObject * args) {
    const char * cfg_file;
    if(!PyArg_ParseTuple(args, "s", &cfg_file))
        return NULL;
    
    if (!ensemble) {
        ensemble = new Ensemble();
    }

    // Load the ensemble
    ensemble->LoadFromCFG(cfg_file);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * add_g2_function(PyObject * self, PyObject * args) {
    double rs, eta;
    if (!PyArg_ParseTuple(args, "dd", &rs, &eta)) {
        cerr << "Error, this function requires rs and eta (double type)" << endl;
        return NULL;
    }

    if (!sym_functs) {
        sym_functs = new SymmetricFunctions();
    }

    sym_functs->AddG2Function(rs, eta);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * add_g4_function(PyObject * self, PyObject * args) {
    double eta, zeta, lambda;
    if (!PyArg_ParseTuple(args, "ddd", &eta, &zeta, &lambda)) {
        cerr << "Error, this function requires eta, zeta and lambda (double type)" << endl;
        return NULL;
    }

    if (!sym_functs) {
        sym_functs = new SymmetricFunctions();
    }

    sym_functs->AddG4Function(eta, zeta, lambda);

    Py_INCREF(Py_None);
    return Py_None;
}

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
}

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
    PyArrayObject * npy_coords, *npy_types;
    PyObject * atoms;
    double * coords;
    int *types;
    int N_atoms;

    if (!PyArg_ParseTuple(args, "OOO", &atoms, &npy_coords, &npy_types)) {
        cerr << "Error, this function requires 2 arguments" << endl;
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

    // Fill the atoms with the correct data
    for (int i = 0; i < N_atoms; ++i) {
        atm->types[i] = types[i];
        for (int j = 0; j <3; ++j) {
            atm->coords[3 * i + j] = coords[3*i+j];
        }
    }

    return Py_BuildValue("");
}

PyObject* get_symmetric_functions_from_atoms(PyObject * self, PyObject * args) {
    PyObject * symf, *atm;
    int N_atoms;
    int N_syms;
    int Nx, Ny, Nz; // The periodic images of the atoms

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "OOiii", &symf, &atm, &Nx, &Ny, &Nz)) {
        cerr << "Error, this function requires 5 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);
    Atoms * atoms = (Atoms*) PyCapsule_GetPointer(atm, NAME_ATOMS);

    N_atoms = atoms->GetNAtoms();
    N_syms = symm_func->GetTotalNSym(atoms->GetNTypes());

    // Allocate the symmfunction arrays
    double * sym_coords = new double[N_atoms * N_syms];

    // Calculate the symmetric coordinates
    symm_func->GetSymmetricFunctions(atoms, Nx, Ny, Nz, sym_coords);

    // Generate the output numpy array
    npy_intp symfunc_dims[2] = {N_atoms, N_syms};
    PyObject* output_array = PyArray_SimpleNewFromData(2, symfunc_dims, NPY_DOUBLE, (void*) sym_coords);

    return Py_BuildValue("O", output_array);
}