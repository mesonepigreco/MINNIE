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
PyObject * atomic_network_load_from_cfg(PyObject * self, PyObject * args);
PyObject * atomic_network_save_to_cfg(PyObject * self, PyObject * args);
PyObject * construct_atoms(PyObject*self, PyObject * args);
PyObject * construct_ensemble(PyObject* self, PyObject * args);
PyObject * set_atoms_coords_type(PyObject * self, PyObject * args);
PyObject * get_symmetric_functions_from_atoms(PyObject * self, PyObject * args);
PyObject * get_symmetric_functions_parameters(PyObject * self, PyObject * args);
PyObject * set_symmetric_functions_parameters(PyObject * self, PyObject * args);
PyObject * set_cutoff(PyObject * self, PyObject * args);
PyObject * get_cutoff(PyObject * self, PyObject * args);
PyObject * set_cutoff_type(PyObject * self, PyObject * args);
PyObject * get_n_sym_functions(PyObject * self, PyObject * args);
PyObject * sym_print_info(PyObject * self, PyObject * args);
PyObject * override_ensemble(PyObject * self, PyObject * args);
PyObject * create_atomic_network(PyObject * self, PyObject * args);
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
    {"SetAtomsCoordsTypes", set_atoms_coords_type, METH_VARARGS, "Set from python the Atoms class attributes"},
    {"GetSymmetricFunctions", get_symmetric_functions_from_atoms, METH_VARARGS, "Get the symmetric functions for the atoms class"},
    {"GetSymmetricFunctionParameters", get_symmetric_functions_parameters, METH_VARARGS, "Get the parameters of the symmetric function."},
    {"SetSymmetricFunctionParameters", set_symmetric_functions_parameters, METH_VARARGS, "Set the parameters of the symmetric function."},
    {"SetCutoffRadius", set_cutoff, METH_VARARGS, "Set the cutoff radius."},
    {"SetCutoffType", set_cutoff_type, METH_VARARGS, "Set the cutoff funciton type."},
    {"GetCutoffTypeRadius", get_cutoff, METH_VARARGS, "Get the cutoff funciton (type and radius)."},
    {"SymPrintInfo", sym_print_info, METH_VARARGS, "Print Info about the symmetric functions"},
    {"GetNSyms", get_n_sym_functions, METH_VARARGS, "Get the number of symmetric functions."},
    {"LoadNNFromCFG", atomic_network_save_to_cfg, METH_VARARGS, "Load the NN from the configuration file"},
    {"SaveNNToCFG", atomic_network_save_to_cfg, METH_VARARGS, "Save the NN into the configuration file"},
    {"CreateEnsembleClass", construct_ensemble, METH_VARARGS, "Create an empty ensemble."},
    {"OvverrideEnsembleIndex", override_ensemble, METH_VARARGS, "Override the i-th structure of the ensemble."},
    {"CreateAtomicNN", create_atomic_network, METH_VARARGS, "Create a new Atomic NN from ensemble"},
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
    PyObject * symFuncs;
    if (!PyArg_ParseTuple(args, "Odd", &symFuncs, &rs, &eta)) {
        cerr << "Error, this function requires The sym function, rs and eta (double type)" << endl;
        return NULL;
    }

    // Retain the pointer to the symmetric function class
    SymmetricFunctions* sym_funcs = (SymmetricFunctions*) PyCapsule_GetPointer(symFuncs, NAME_SYMFUNC);
    cout << "HERE INSIDE" << endl;
    sym_functs->AddG2Function(rs, eta);
    cout << "HERE INSIDE" << endl;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * add_g4_function(PyObject * self, PyObject * args) {
    double eta, zeta;
    int lambda;
    PyObject * symFuncs;
    if (!PyArg_ParseTuple(args, "Oddi", &eta, &zeta, &lambda)) {
        cerr << "Error, this function requires sym fuunctions eta, zeta(double type) and lambda (int)" << endl;
        return NULL;
    }

    SymmetricFunctions* sym_funcs = (SymmetricFunctions*) PyCapsule_GetPointer(symFuncs, NAME_SYMFUNC);


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
    Ensemble* ens = new Ensemble(Natoms, Nconfigs);

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
    int index, g2or4;

    // Parse the python arguments
    if (!PyArg_ParseTuple(args, "O", &symf)) {
        cerr << "Error, this function requires 1 arguments" << endl;
        cerr << "Error on file " << __FILE__ << " at line " << __LINE__ << endl;
        return NULL;
    }

    // Get the correct C++ data types
    SymmetricFunctions* symm_func = (SymmetricFunctions*) PyCapsule_GetPointer(symf, NAME_SYMFUNC);

    int n2, n4;
    n2 = symm_func->get_n_g2();
    n4 = symm_func->get_n_g4();

    return Py_BuildValue("ii", n2, n4);
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
