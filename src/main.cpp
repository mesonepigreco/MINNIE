#include <stdio.h>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <libconfig.h++>
#include <algorithm>
#include "network.hpp"

using namespace std;
using namespace libconfig;


// Load the training data from file
void load_dataset(char * fname, Dataset * &data,
		  Dataset * &test_set, bool Load_config_only = false) ;

// Test if all the required voices exists
void TestCFG(char *);

// The operations to be performed in the training or predict mode
void TrainMode(char * config);
void PredictionMode(char * config);



// -------------------------------------------- BEGINING OF THE MAIN ----------------------------------------------------
int main(int argc, char ** argv) {
  Config cfg;

  
  // Get as only argument the configuration file
  if (argc != 2) {
    cerr << "Error, you must specify a configuration file." << endl;
    cerr << "Only one argument is required." << endl;
    cerr << "Aborting." << endl;
    return EXIT_FAILURE;
  }

  // Read the input file
  // Test if the required flag are present
  TestCFG(argv[1]);

  // Load the data
  cfg.readFile(argv[1]);

  // Prepare the training of the neural network

  // Chose the mode
  string mode = cfg.lookup("mode");
  
  // * * * * * * * * * *  TRAINING MODE  * * * * * * * * * * * * 
  if ( mode == "train" ) {
    TrainMode(argv[1]);
  //  * * * * * * * * *  PREDICTION MODE  * * * * * * * * * * *
  } else if ( mode == "predict") {
    PredictionMode(argv[1]);
  }

  cout << endl;
  cout << "DONE" << endl;
  return 0;
}

// ---------------------------------------------------- END OF MAIN ----------------------------------------------------



/*
 *
 *
 *
 * OTHER FUNCTIONS
 *
 *
 *
 */



void TrainMode(char * config_name) {
  Config cfg;
  Dataset *data, *test_set;

  // Read the configuration file
  cfg.readFile(config_name);
  
  // Setup the verbosity level
  int verbosity = 1;
  try{ verbosity = cfg.lookup("verbosity"); }
  catch (const SettingException &errortype){}

  if (verbosity >= 1) {
    cout << "Reading network information." << endl;
  }
  
  // Read the structure of the neural network
    
  // Load the dataset as specified in the configuration file
  load_dataset(config_name, data, test_set);

  // Build the initial neural network
  NeuralNetwork *NN;
  string nn_file;

  const Setting& root = cfg.getRoot();
  const Setting& NN_setting = root["NeuralNetwork"];

  // If an input file is given, load the network from there
  if (cfg.lookupValue("inputfile", nn_file)){
    cout << "Neural network read from: " << nn_file << endl;
    NN = new NeuralNetwork(nn_file);
  } 
  // Otherwise, build it from scratch
  else {
    cout << "Neural network start from scratch." << endl;
    int n_incoming, n_outcoming, n_hidden_layers;
    // Check if the input file is correctly formatted
    try {
	NN_setting.lookupValue("n_input", n_incoming);
	NN_setting.lookupValue("n_output", n_outcoming);
	NN_setting.lookupValue("n_hidden", n_hidden_layers);


	int nodes_hl;
	nodes_hl = NN_setting.lookup("n_nodes_hidden");
	NN = new NeuralNetwork(n_incoming, n_outcoming, n_hidden_layers, 
			       NULL, nodes_hl);
    } catch (const SettingException &error) {
      try {
	// Load the number of hidden nodes from the array
	const Setting &n_nodes_setting = NN_setting["list_nodes_hidden"];
	
	// Check the array length
	if (n_nodes_setting.getLength() != n_hidden_layers) {
	  cerr << "Error, the lenght of key 'list_nodes_hidden' must match 'n_hidden'" << endl;
	  cerr << "Abort" << endl;
	  exit( EXIT_FAILURE);
	}
	
	int * nodes_hl;
	nodes_hl = (int*) malloc(sizeof(int) * n_hidden_layers);
	for (int i = 0; i < n_nodes_setting.getLength(); ++i) 
	  nodes_hl[i] = n_nodes_setting[i];
	
	NN = new NeuralNetwork(n_incoming, n_outcoming, n_hidden_layers,
			       nodes_hl);
      
      } catch (const SettingException &error) {
	cerr << "Error, check setting " << error.getPath() << " carefully" << endl;
	exit(EXIT_FAILURE);
      }
    }
  }

  // Read learning rate, step and inertia for the training
  double learning_rate = 0.05;
  int max_steps = 1000;
  double learning_inertia;
  if (NN_setting.lookupValue("learning_rate", learning_rate)) {
    if (verbosity >= 1) cout << "Setting learning_rate to " << learning_rate
			     << endl;
    NN-> set_learning_step(learning_rate);
  } else if (verbosity >= 1) {
    cout << "learning_rate key not found (it must be a double). Setting learning_rate = "
	 << learning_rate << endl;
  }
  
  if (NN_setting.lookupValue("max_steps", max_steps)) {
    if (verbosity >= 1) cout << "Setting max_steps to " <<
			  max_steps << endl;
    NN->set_max_steps(max_steps);
  } else if (verbosity >= 1) {
    cout << "max_steps key not found. Setting max_step = " << max_steps << endl;
  }
    
  if (NN_setting.lookupValue("learning_inertia", learning_inertia)) {
    if (verbosity >= 1) cout << "Setting learning_inertia to " <<
			  learning_inertia << endl;
    NN->set_learning_inertia(learning_inertia);
  } else if (verbosity >= 1) {
    cout << "learning_inertia key not found. Setting learning_inertia = "
	 << learning_inertia << endl;
  }
  
  // Perform the training
  cout << endl;
  if (verbosity >= 1) {
    cout << "Traning the network..." << endl;
    cout << "Total number of biases: " << NN->get_nbiases() << endl;
    cout << "Total number of sinapsis: " << NN->get_nsinapsis() << endl;
  }

  // Setup the training algorithm
  string train_alg;
  double t_start, t_decay;
  if (NN_setting.lookupValue("training_algorithm", train_alg)) {
    if (train_alg == "annealing-force") {
      // Check if all the important variables are specified
      if (! (NN_setting.lookupValue("t_start", t_start) && NN_setting.lookupValue("t_decay", t_decay))) {
	cerr << "Please specify both t_start and t_decay with the correct type (double)." << endl;
	exit(EXIT_FAILURE);
      }

      if (t_decay <= 0 || t_decay >= 1) {
	cerr << "Error: t_decay must in the (0, 1) set" << endl;
	exit(EXIT_FAILURE);
      }

      // Setup the beta direction
      double beta_dir;
      int acc_rate;
      if (NN_setting.lookupValue("beta_direction", beta_dir))
	NN->set_beta_direction(beta_dir);
      else 
	cout << "beta_direction non correctly specified (dobule). Setted as default = " << NN->get_beta_direction() << endl;
	
      
      if (NN_setting.lookupValue("accept_rate", acc_rate))
	NN->set_accept_rate(acc_rate);
      else
	cout << "accept_rate not correctly specified (int). Setted as default = " << NN->get_accept_rate() << endl;
    }
    
    NN->set_training_algorithm(train_alg, t_decay, t_start);
  }
  else if (verbosity >= 1) {
    cout << "No training algorithm setted. Using the default 'sdes'" << endl;
    cout << "Warning: training variables referring to other algorithms will be igonred" << endl;
  }

  // Setup the training batch
  int batch_size, bootstrap_size;
  batch_size = data->get_ndata();
  
  // Check if batch size and bootstrap exists
  if (cfg.lookupValue("batch_size", batch_size)) {
    if (cfg.lookupValue("total_size", bootstrap_size)) {
      cout << "Bootstrap up to " << bootstrap_size << endl;
      data->bootstrap(bootstrap_size);
    }
    else {
      cout << "Error, if you want to perform the batch minimization, specify the total_size."<<  endl;
      cerr << "Error, if you want to perform the batch minimization, specify the total_size."<<  endl;
      exit(EXIT_FAILURE);
    }
  } else {
    cout <<"Warning, no batch_size found. The whole training set will be used."<<endl;
  }
	  
  cout << endl;
  
  // Test the gradient
  //NN->TestGradient(data, loss_minsquare, gradient_minsquare, 3,0);

  // Train the network
  NN->TrainNetwork(data, test_set, batch_size, loss_minsquare, gradient_minsquare);

  // Network trained
  cout << endl;
  cout << "Traning done!" << endl;

  cout << "Saving the neural network" << endl;
  string save_fname;
  if (cfg.lookupValue("outputfile", save_fname)) {
    cout << "File : " << save_fname << endl;
    NN-> Save(save_fname);
  }
}





void PredictionMode(char * config_name) {
  Config cfg;

  
  // Read the configuration file
  cfg.readFile(config_name);
  
  // Read the neural network from file
  string nn_file;
  if (! cfg.lookupValue("inputfile", nn_file)) {
    cerr << "Error: in mode 'predict' a correct inputfile for the network is requested." << endl;
    exit(EXIT_FAILURE);
  }
  
  // Load the network from the specified file
  NeuralNetwork *NN = new NeuralNetwork(nn_file);
  
  // Load the data to be predicted
  Dataset *data;
  load_dataset(config_name, data, data, true);
  
  // Write the energy
  int n_atoms = cfg.lookup("n_atoms");
  string datadir, energy_fname, root_conf, root_forces;
  cfg.lookupValue("datadir", datadir);
  cfg.lookupValue("energy_fname",energy_fname);
  
  // Prepare the file for the writing
  ofstream en_file, force_file;
  en_file.open( datadir + "/" + energy_fname);
  
  double energy, dumb = 1;
  double * forces = (double*) calloc(sizeof(double), 3 * n_atoms);

  // Check if the auxiliary force and energy must be added
  string aux_energy_name, aux_force_name;
  bool add_forces = false, add_energy = false;
  ifstream aux_energy, aux_force;

  // Open the force file
  add_forces = cfg.lookupValue("aux_forces", aux_force_name);
  add_energy = cfg.lookupValue("aux_energies", aux_energy_name); 
  if ( add_forces != add_energy ) {
    cout << "ERROR: both aux_forces and aux_energies must be provided." << endl;
    cerr << "ERROR: both aux_forces and aux_energies must be provided." << endl;
    exit(EXIT_FAILURE);
  }

  // Open the auxiliary energy file if requested
  if (add_energy) {
    aux_energy.open(datadir + "/" + aux_energy_name);
    if (! aux_energy.good() ) {
      cerr << "Error: the specified file for the energies does not exist." <<endl;
      cerr << "       filename : " << datadir + "/" + energy_fname << endl;
      cout << "Error: the specified file for the energies does not exist." <<endl;
      cout << "       filename : " << datadir + "/" + energy_fname << endl;
      exit (EXIT_FAILURE);
    }
  }
  
  for (int i = 0; i < data->get_ndata(); ++i) {
    // Predict the features
    NN->PredictFeatures(1, data->get_feature(i), &energy, true);
    // Add the aux energy
    if (add_energy) {
      double tmp;
      string line;
      getline(aux_energy, line);
      replace(line.begin(), line.end(), 'D', 'e');
      istringstream in(line);
      in >> tmp;
      energy += tmp;
    }
       
    en_file << scientific << energy << endl;
    
    // Compute the forces
    if (cfg.lookupValue("root_forces", root_forces)) { 
      NN->GetForces(forces, true);
      force_file.open(datadir + "/" + root_forces + to_string(i+1) + ".dat");

      // If necessary load the auxiliary forces
      if (add_forces) {
	aux_force.open(datadir + "/" + aux_force_name + to_string(i+1) + ".dat");
	if (! aux_force.good() ) {
	  cerr << "Error: force file " << datadir + "/" + aux_force_name  + to_string(i+1) + ".dat" << " not found." << endl;
	  cout << "Error: force file " << datadir + "/" + aux_force_name  + to_string(i+1) + ".dat" << " not found." << endl;
	  exit(EXIT_FAILURE);
	}
      }
      
      for (int j = 0; j < n_atoms; ++j) {
	// If necessary add the auxiliary forces
	if (add_forces) {
	  string line;
	  double fx, fy, fz;
	  getline(aux_force, line);
	  replace(line.begin(), line.end(), 'D', 'e');
	  istringstream in(line);
	  in >> fx >> fy >> fz;
	  forces[3*j] += fx;
	  forces[3*j + 1] += fy;
	  forces[3*j + 2] += fz;
	}
	
	force_file << scientific << forces[3*j] << "\t"
		   << scientific << forces[3*j + 1] << "\t"
		   << scientific << forces[3*j + 2] << endl;

      }
      force_file.close();

      if (add_forces)
	aux_force.close();
    } else if (i == 0) {
      cout << "No force computation required." << endl;
      cout << "If you want to compute forces, please specify" << endl;
      cout << "the 'root_forces' key in the configuration file." << endl;
      cerr << "Warning: no output on forces." << endl;
    }
  }
  
  // Done
  cout << "Total energy calculation termined." << endl;
}



void TestCFG(char * filename) {
  Config cfg;
  
  try {
    cfg.readFile(filename);
  } catch (const FileIOException &fioex) {
    cerr << "I/O error whle reading file." << endl;
    exit( EXIT_FAILURE);
  } catch (const ParseException &pex) {
    cerr << "Error while parsing the configuration file:" << endl;
    cerr << "File : " << pex.getFile() << " => Line: " << pex.getLine() << endl;
    cerr << pex.getError() << endl;
    exit( EXIT_FAILURE);
  }

  try {
    cfg.lookup("datadir");
    cfg.lookup("n_configs");
    cfg.lookup("root_name");
    cfg.lookup("n_atoms");
    cfg.lookup("energy_fname");
    cfg.lookup("NeuralNetwork.n_input");
    cfg.lookup("NeuralNetwork.n_output");
    cfg.lookup("NeuralNetwork.n_hidden");
    cfg.lookup("mode");
    
    // CONTINUE ...
  } catch (const SettingTypeException &errortype) {
    cerr << "Error, the setting '" << errortype.getPath() << "' does not match the type." << endl;
    exit (EXIT_FAILURE);
  } catch (const SettingNotFoundException &errorfound) {
    cerr << "Error, the setting '" << errorfound.getPath() << "' is not present." << endl;
    exit (EXIT_FAILURE);
  }
}


void load_dataset(char* fname, Dataset *&data, Dataset *&test_set, bool Load_config_only) {
  Config cfg;
  
  string datadir, energy_fname;
  string root_name;
  string root_forces;
  int n_configs;
  int n_atoms;
  int n_test = 0;
  double eq_energy = 0;

  
  cfg.readFile(fname);

  
  cfg.lookupValue("datadir", datadir);
  cfg.lookupValue("n_configs", n_configs);
  cfg.lookupValue("root_name", root_name);
  cfg.lookupValue("n_atoms", n_atoms);
  cfg.lookupValue("energy_fname", energy_fname);
  cfg.lookupValue("eq_energy", eq_energy) ;

  if (cfg.lookupValue("n_test", n_test)) {
    if (n_test >= n_configs || n_test < 0) {
      cerr << "Error: the number of configuration in the test set" <<endl
	   << "       must be equal or lower the training set size." << endl;
      exit(EXIT_FAILURE);
    }

    // Load the test_set
    test_set = new Dataset(n_test, 3* n_atoms, 1);
  } else {
    cout << "Warning: no test set provided." << endl;
    cerr << "Warning: no test set provided." << endl;
  }
  
  // Check if the data dir ends with the line backslash delete it
  if (datadir.back() == '/') 
    datadir.pop_back();
  
  // Prepare the training set
  data = new Dataset (n_configs - n_test, 3 * n_atoms, 1);
  

  // Load all the dataset
  int i, j;
  string line;

  for (i = 0; i < n_configs; ++i) {
    ifstream u_file (datadir + "/" + root_name + to_string(i) + ".dat");

    

    for (j = 0; j < n_atoms; ++j) {
      getline(u_file, line);
      istringstream in(line);

      // Read the line parsing the three coordinates
      if (i < n_configs - n_test) {
	in >> data->features[3*n_atoms*i + 3*j]
	   >> data->features[3*n_atoms*i + 3*j + 1]
	   >> data->features[3*n_atoms*i + 3*j + 2];
      }
      else {
	in >> test_set->features[3*n_atoms*(i - n_configs + n_test) + 3*j]
	   >> test_set->features[3*n_atoms*(i - n_configs + n_test) + 3*j + 1] 
	   >> test_set->features[3*n_atoms*(i - n_configs + n_test) + 3*j + 2];
	
      }	
    }

    u_file.close();

  }

  // Load the energies
  if (!Load_config_only) {
    ifstream en_file( datadir + "/" + energy_fname);
    ifstream aux_energies;
    string aux_energy_fname;
    bool sub_energies = false;

    if (! en_file.good() ) {
      cerr << "Error: the specified file for the energies does not exist." <<endl;
      cerr << "       filename : " << datadir + "/" + energy_fname << endl;
      cout << "Error: the specified file for the energies does not exist." <<endl;
      cout << "       filename : " << datadir + "/" + energy_fname << endl;
      exit (EXIT_FAILURE);
    }
    
    // If the auxiliary energy is given in input, use it
    if (cfg.lookupValue("aux_energies", aux_energy_fname)) {
      sub_energies = true;
      aux_energies.open(datadir + "/" + aux_energy_fname);
      if (! aux_energies.good() ) {
	cerr << "Error: the specified file for the scha energies does not exist." <<endl;
	cerr << "       filename : " << datadir + "/" + aux_energy_fname << endl;
	cout << "Error: the specified file for the scha energies does not exist." <<endl;
	cout << "       filename : " << datadir + "/" + aux_energy_fname << endl;
	exit (EXIT_FAILURE);
      }
    } else {
      cout << "Warning: no aux energies setted. It could be difficult to train the network." <<endl;
      cerr << "Warning: no aux energies setted. It could be difficult to train the network." <<endl;
    }
    for (i = 0; i < n_configs; ++i) {
      getline(en_file, line);

      // Format the string in a C++ fashion
      replace(line.begin(), line.end(), 'D', 'e');
      
      istringstream in(line);
      if (i < n_configs - n_test) {
	in >> data->values[i];
	data->values[i] -= eq_energy;

	// Check if the auxenergy is given
	if (sub_energies) {
	  getline(aux_energies, line);
	  
	  // Format the string in a C++ fashion
	  replace(line.begin(), line.end(), 'D', 'e');

	  istringstream aux (line);
	  double aux_en;
	  aux >> aux_en;
	  data->values[i] -= aux_en;
	  if (i == 0 && eq_energy != 0) {
	    cout << "Warning: both eq_energy and aux_energies setted." << endl;
	    cout << "         please, check if that is what you really want." << endl;
	    cerr << "Warning: both eq_energy and aux_energies setted." << endl;
	    cerr << "         please, check if that is what you really want." << endl;
	  }
	}
      }
      else {
	in >> test_set->values[i - n_configs + n_test];
	test_set->values[i - n_configs + n_test] -= eq_energy;
	
	if (sub_energies) {
	  getline(aux_energies, line);
	  
	  // Format the string in a C++ fashion
	  replace(line.begin(), line.end(), 'D', 'e');

	  istringstream aux (line);
	  double aux_en;
	  aux >> aux_en;
	  test_set->values[i - n_configs + n_test] -= aux_en;
	}
      }
    }
    en_file.close();
    if (sub_energies)
      aux_energies.close();

    // Check the consistency
    if (cfg.lookupValue("root_forces", root_forces)) {
      data->has_forces = true;
      test_set->has_forces = true;

      // Check if the forces must be read
      string aux_forces_name;
      bool sub_forces = false;
      if (cfg.lookupValue("aux_forces", aux_forces_name)) {
	sub_forces = true;
      }

      // Check the consistency with energy
      if (sub_energies != sub_forces) {
	cerr << "Error: if you want to train with forces and with an energy/force correction," << endl;
	cerr << "       you must specify both 'aux_energies' and 'aux_forces'." << endl;
	cout << "Error: if you want to train with forces and with an energy/force correction," << endl;
	cout << "       you must specify both 'aux_energies' and 'aux_forces'." << endl;
	exit(EXIT_FAILURE);
      }
	
      for (i = 0; i < n_configs; ++i) {
	ifstream f_file (datadir + "/" + root_forces + to_string(i+1) + ".dat");
	if (! f_file.good() ) {
	  cerr << "Error: force file " << datadir + "/" + root_forces + to_string(i+1) + ".dat" << " not found." << endl;
	  cout << "Error: force file " << datadir + "/" + root_forces + to_string(i+1) + ".dat" << " not found." << endl;
	  exit(EXIT_FAILURE);
	}

	// Check if the scha forces file is present
	ifstream aux_forces;
	if (sub_forces) {
	  aux_forces.open(datadir + "/" + aux_forces_name + to_string(i+1) + ".dat");
	  if (! aux_forces.good() ) {
	    cerr << "Error: force file " << datadir + "/" + aux_forces_name  + to_string(i+1) + ".dat" << " not found." << endl;
	    cout << "Error: force file " << datadir + "/" + aux_forces_name  + to_string(i+1) + ".dat" << " not found." << endl;
	    exit(EXIT_FAILURE);
	  }
	}
	
	for (j = 0; j<  n_atoms; ++j) {
	  getline(f_file, line);

	  // Format the string in a C++ fashion
	  replace(line.begin(), line.end(), 'D', 'e');

	  istringstream in(line);
	  if (i < n_configs - n_test) {
	    in >> data->forces[3 * n_atoms *i + 3*j]
	       >> data->forces[3 * n_atoms *i + 3*j + 1]
	       >> data->forces[3 * n_atoms *i + 3*j + 2];	    
	    // If the auxiliary forces must be read subtract them to the read ones
	    if (sub_forces) {
	      getline(aux_forces, line);

	      // Format the string in a C++ fashion
	      replace(line.begin(), line.end(), 'D', 'e');

	      istringstream aux(line);
	      double fx, fy, fz;
	      aux >> fx >> fy >> fz;

	      // Subtract the force
	      data->forces[3*n_atoms*i + 3*j] -= fx;
	      data->forces[3*n_atoms*i + 3*j + 1] -= fy;
	      data->forces[3*n_atoms*i + 3*j + 2] -= fz;
	    }
	  }
	  else {
	    in >> test_set->forces[3*n_atoms*(i - n_configs + n_test) + 3*j]
	       >> test_set->forces[3*n_atoms*(i - n_configs + n_test) + 3*j + 1] 
	       >> test_set->forces[3*n_atoms*(i - n_configs + n_test) + 3*j + 2];

	    // Subtract the auxiliary forces from the test set
	    if (sub_forces) {
	      getline(aux_forces, line);

	      // Format the string in a C++ fashion
	      replace(line.begin(), line.end(), 'D', 'e');

	      istringstream aux(line);
	      double fx, fy, fz;
	      aux >> fx >> fy >> fz;

	      // Subtract the force
	      test_set->forces[3*n_atoms*(i - n_configs + n_test) + 3*j] -= fx;
	      test_set->forces[3*n_atoms*(i - n_configs + n_test) + 3*j + 1] -= fy;
	      test_set->forces[3*n_atoms*(i - n_configs + n_test) + 3*j + 2] -= fz;
	      
	    }
	  }
	}
	f_file.close();

	if (sub_forces) aux_forces.close();
      }
    }
  }
}



