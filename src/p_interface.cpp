#include <python2.7/Python.h>
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


static PyMethodDef Methods[] = {
    {"LoadEnsemble", load_ensemble_from_dir, METH_VARARGS, "Load the ensemble from a directory"},
    {"PrintConfig", print_configuration, METH_VARARGS, "Print the given configuration on stdout"},
    {"LoadEnsembleFromCFG", load_ensemble_from_cfg, METH_VARARGS, "Load the ensemble from a configuration file."},
    {"AddSymG2", add_g2_function, METH_VARARGS, "Add a G2 symmetry function"},
    {"AddSymG4", add_g4_function, METH_VARARGS, "Add a G4 symmetry function"},
    {"LoadSymmetricFunctionsFromCFG", load_symmetric_functions_from_cfg, METH_VARARGS, "Load a set of symmetric functions from a configuration file."},
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