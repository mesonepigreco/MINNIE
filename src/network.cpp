#include "network.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <cmath>
#include <libconfig.h++>
#include "utils.hpp"

using namespace libconfig;
#define ZERO 1e-12

// The dataset building function
Dataset::Dataset(int ndata, int nfeatures, int nvalues,
		 double * pfeatures, double * pvalues) {
  N_data = ndata;
  N_features = nfeatures;
  N_values = nvalues;

  has_forces = false;

  // Allocate the memory for the forces
  forces = (double *) calloc(sizeof(double), N_features * ndata);

  cout << "Prova init data" << endl;
  
  // Setup the pointers
  if (pfeatures != NULL)
    features = pfeatures;
  else
    features = (double*) calloc(sizeof(double), N_features * N_data);
  if (pvalues != NULL)
    values = pvalues;
  else
    values = (double*)calloc(sizeof(double), N_values * N_data);
}



Dataset::~Dataset(){
  free(features);
  free(forces);
  free(values);
}

int Dataset::get_ndata(void) {return N_data;};
int Dataset::get_nfeatures(void) {return N_features;};
int Dataset::get_nvalues(void) {return N_values;};

double * Dataset::get_feature(int i){return features + i * N_features;};
double * Dataset::get_values(int i){return values + i * N_values;};
double * Dataset::get_forces(int i){return forces + i * N_features;};


// Bootstrap
void Dataset::bootstrap(int new_dim) {
  // Prepare a new feature and values array
  double * new_features = (double*) malloc(sizeof(double)* new_dim * N_features);
  double * new_values = (double*) malloc(sizeof(double)* new_dim * N_values);


  // Start the shuffling process
  int i, j, index;
  for (i = 0; i < new_dim; ++i) {
    // Extact the random index
    index = rand() % N_data;

    // Copy the features and values of the given index
    for (j = 0; j < N_features; ++j)
      new_features[ N_features * i + j] = get_feature(index)[j];
 
    for (j = 0; j < N_values; ++j)
      new_values[ N_values * i + j] = get_values(index)[j];
  }


  // free the memory
  free(features);
  free(values);

  // Overplace the old variables with the new one
  features = new_features;
  values = new_values;

  // Override the data dimension
  N_data = new_dim;
}



// ---------------------- NEURAL NETWORK --------------------------
NeuralNetwork::NeuralNetwork(int ninc, int nout, int n_hidden,
			     int * nodes_per_hl, int nodes_hl) {
  // Setup variables
  is_trained = false;
  N_hidden_layers = n_hidden;

  // Setup the default algorithm  
  beta_direction = -1;
  energy_beta_equal = true;
  accept_rate = 10;
  t_decay = 0;
  t_start = 1;
  set_training_algorithm("sdes");

  verbosity_level = 1;
  data_dir = ".";
  
  // Default value for the learning rates
  learning_rate = 0.1;
  learning_inertia = 0;

  N_nodes.push_back(ninc);

  // Setup the initial layer of neurons
  int i, j;
  for (i = 0; i < ninc; ++i)
    neurons.push_back(0);

  // Setup all the grid
  for (i = 0; i < n_hidden; ++i) {
    if (nodes_per_hl) 
      N_nodes.push_back(nodes_per_hl[i]);
    else
      N_nodes.push_back(nodes_hl);
    
    // Prepare the biases
    for (j = 0; j < N_nodes.back(); ++j) {
      biases.push_back( (rand() / (double)RAND_MAX) * 2 - 1);
      neurons.push_back(0);
    }

    // Prepare the sinapsis
    for (j = 0; j < N_nodes.at(i) * N_nodes.at(i+1); ++j)
      sinapsis.push_back( (rand() / (double)RAND_MAX) * 2 - 1);
  }
  
  N_nodes.push_back(nout);
  
  // Prepare the last biases
  for (j = 0; j < nout; ++j) {
    biases.push_back( (rand() / (double)RAND_MAX) * 2 - 1);
    neurons.push_back(0);
  }	  

  // Prepare the last sinapsis
  for (j = 0; j < N_nodes.at(n_hidden) * nout; ++j)
    sinapsis.push_back( (rand() / (double)RAND_MAX) * 2 - 1);


  // Initialize the input/output linear transformation
  input_mean = (double*) calloc(sizeof(double), N_nodes.at(0));
  input_sigma = (double*) calloc(sizeof(double), N_nodes.at(0));
  output_sigma = (double*) calloc(sizeof(double), get_npredictions());
  output_mean = (double*) calloc(sizeof(double), get_npredictions());
}


NeuralNetwork::~NeuralNetwork() {
  // PASS FOR NOW
}

int NeuralNetwork::get_neuron_index(int layer, int index) {
  int i, ret = 0;
  for (i = 0; i < layer; ++i)
    ret += N_nodes.at(i);
  return ret + index;
}

int NeuralNetwork::get_biases_index(int layer, int index) {
  return get_neuron_index(layer, index) - N_nodes.at(0);
}

/*
 * This get_sinapsis_starting_layer the neuron backward index, used in the backpropagation
 * algorithm
 */
// int NeuralNetwork::get_inverse_neuron_index(int layer, int index) {
//   int i, ret = 0;

//   for (i = N_hidden_layers + 1; i > layer; --i) 
//     ret += N_nodes.at(i);
//   return ret + index;
// }

int NeuralNetwork::get_sinapsis_index(int starting_layer, int starting_index, int end_index) {
  int i, ret = 0;
  for (i = 1; i < starting_layer; ++i) {
    ret += N_nodes.at(i-1) * N_nodes.at(i); 
  }

  return ret + N_nodes.at(starting_layer) * end_index + starting_index;
}

int NeuralNetwork::get_sinapsis_starting_layer(int index) {
  int i;
  int layer = 0;
  int shift = 0;
  while (shift + N_nodes.at(layer)* N_nodes.at(layer + 1) < index) {
    shift += N_nodes.at(layer) * N_nodes.at(layer + 1);
    layer += 1;
    if (layer > N_hidden_layers) {
      cerr << "Error, the index given exceeds the number of hidden layers" << endl;
      throw "";
    }
  }
  return layer;
}

int NeuralNetwork::get_npredictions() {
  return N_nodes.at( N_nodes.size() -1 );
}


void NeuralNetwork::TrainNetwork(Dataset * training_set, Dataset * test_set, int batch_size,
				 double (*loss_function) (int, const double * ,const double *),
				 void(*gradient_loss) (int, const double *, const double *, double *)){

  
  // Prepare the gradient and the predictions
  
  double * grad_biases, *grad_sinapsis;
  double * grad_biases_old, *grad_sinapsis_old;
  double * predictions, * grad_loss;
  double * predicted_forces;
  double * old_biases, *old_sinapsis;

  grad_biases = (double*) calloc(sizeof(double), biases.size());
  grad_biases_old = (double*) calloc(sizeof(double), biases.size());
  grad_sinapsis = (double*) calloc(sizeof(double), sinapsis.size());
  grad_sinapsis_old = (double*) calloc(sizeof(double), sinapsis.size());
  predictions = (double*) calloc(sizeof(double), get_npredictions() * batch_size);
  grad_loss = (double *) calloc(sizeof(double), get_npredictions());

  
  double * r_features = (double*) malloc(sizeof(double) * batch_size * get_npredictions());

  if (train_with_forces) {
    predicted_forces = (double*) calloc(sizeof(double), N_nodes.at(0) * batch_size);
    old_biases = (double*) calloc(sizeof(double), biases.size());
    old_sinapsis = (double*) calloc(sizeof(double), sinapsis.size());
  }

  int ka, i, start_config_index, config_index;
  double g_biases, g_sinapsis, loss, loss_forces;
  double new_loss, new_loss_forces;
  double g_biases_old, g_sinapsis_old, cos_alpha;


  // Setup input and output scaling
  for (int i = 0; i < N_nodes.at(0) ; ++i) {
    double sum = 0, sum2 = 0;
    for (int j = 0; j < training_set->get_ndata(); ++j) {
      sum += training_set->get_feature(j)[i];
      sum2 += training_set->get_feature(j)[i] * training_set->get_feature(j)[i];
    }
    input_mean[i] = sum / (double) training_set->get_ndata();
    input_sigma[i] = sum2 / (double) training_set->get_ndata() - sum* sum / (double) (training_set->get_ndata() * training_set->get_ndata() );
    input_sigma[i] = sqrt(input_sigma[i]);

    // Rescale the training set and the test set
    for (int j = 0; j < training_set->get_ndata(); ++j) {
      training_set->get_feature(j)[i] = ( training_set->get_feature(j)[i] - input_mean[i]) / input_sigma[i];
    }
    for (int j = 0; j < test_set->get_ndata(); ++j) {
      test_set->get_feature(j)[i] = ( test_set->get_feature(j)[i] - input_mean[i]) / input_sigma[i];
    }
  }
  for (int i = 0; i < get_npredictions() ; ++i) {
    double sum = 0, sum2 = 0;
    for (int j = 0; j < training_set->get_ndata(); ++j) {
      sum += training_set->get_values(j)[i];
      sum2 += training_set->get_values(j)[i] * training_set->get_values(j)[i];
    }
    output_mean[i] = sum / (double) training_set->get_ndata();
    output_sigma[i] = sum2 / (double) training_set->get_ndata() - sum* sum / (double) (training_set->get_ndata() * training_set->get_ndata() );
    output_sigma[i] = sqrt(output_sigma[i]);

    // Rescale the training set and the test set
    for (int j = 0; j < training_set->get_ndata(); ++j) {
      training_set->get_values(j)[i] = ( training_set->get_values(j)[i] - output_mean[i]) / output_sigma[i];
    }
    for (int j = 0; j < test_set->get_ndata(); ++j) {
      test_set->get_values(j)[i] = ( test_set->get_values(j)[i] - output_mean[i]) / output_sigma[i];
    }    
  }


    
  // Check if the training algorithm is compatible with the dataset
  if (training_algorithm.find("force") != string::npos) {
    if (! (training_set->has_forces && test_set->has_forces)) {
      cerr << "Error: the given training algorithm requires that the datasets contain" << endl;
      cerr << "       information about the forces." << endl;
      cout << "Error: the given training algorithm requires that the datasets contain" << endl;
      cout << "       information about the forces." << endl;
      exit(EXIT_FAILURE);
    }
  }
    
  

  accept_rate = 0;
  start_config_index = 0;
  cout << "Start the training!" << endl;
  for (ka = 0; ka < n_step_max; ++ka) {
    // Start the training ----

    printf(" ----------- TRAINING STEP ----------\n");
    printf(" ka = %d\n", ka);

    // clear the gradient and copy the current gradient into the old one
    for (i = 0; i < biases.size(); ++i) {
      grad_biases_old[i] = grad_biases[i];
      grad_biases[i] = 0;
    }
    for (i = 0; i < sinapsis.size(); ++i) {
      grad_sinapsis_old[i] = grad_sinapsis[i];
      grad_sinapsis[i] = 0;
    }
    
    //cout << "Gradient cleaned." << endl;

    // Cycle over the batchs
    loss = 0;
    loss_forces = 0;
    for (i = 0; i < batch_size; ++i) {
      // Setup the configuration index
      config_index = (start_config_index + i) % training_set->get_ndata();

      // Prepare the prediction for the current configuration
      PredictFeatures(1, training_set->get_feature(config_index), predictions + get_npredictions()*i);
      
      // Prepare the gradient for the given configuration
      gradient_loss(get_npredictions(), predictions + get_npredictions()*i,
		    training_set->get_values(config_index), grad_loss);
      StepDescent(grad_loss, grad_biases, grad_sinapsis);

      // If the forces are required for the training, compute them
      if (train_with_forces) {
	GetForces(predicted_forces + N_nodes.at(0) * i);
	loss_forces += loss_function(N_nodes.at(0), predicted_forces + N_nodes.at(0) * i,
				     training_set->get_forces(config_index)) / (batch_size * N_nodes.at(0));
      }
      
      // Compute the loss for the current configuration
      loss += loss_function(get_npredictions(), predictions + get_npredictions() * i,
			    training_set->get_values(config_index)) / batch_size;
    }

    
    // Print the correlation coefficient (energy)
    double batch_corr, batch_corr_error;
    for (i = 0; i < batch_size; ++i) {
      config_index = (start_config_index + i) % training_set->get_ndata();
      for (int j = 0; j < get_npredictions(); ++j)
	r_features[get_npredictions()*i + j] = training_set->get_feature(config_index)[j];
    }
    batch_corr = pearson_correlation(batch_size * get_npredictions(), predictions,
				     r_features, &batch_corr_error);
    printf ("Pearson of the batch (training) = %.8e +- %.8e\n", batch_corr, batch_corr_error);

    // Check if the verbosity is >= 2. In this case print on a file the batch predicted and real features.
  //   if (verbosity_level >= 2) {
  //     ofstream of_features;
  //     of_features.open(data_dir + "/" + "features_" + to_string(ka) + ".dat");
  //     of_features << "# for i in n_features : ... real [i] predicted[i] ..." << endl;
  //     for (i = 0; i < get_npredictions(); ++i)
	// of_features << 
  //   }

    // Normalize the gradient
    for (i = 0; i< biases.size(); ++i)
      grad_biases[i] /= batch_size;
    for (i = 0; i< sinapsis.size(); ++i)
      grad_sinapsis[i] /= batch_size;


    // Compute the optimal minimization step with the line minimization
    cos_alpha = 0;
    g_biases_old = g_biases;
    g_sinapsis_old = g_sinapsis;
    g_biases = 0;
    g_sinapsis = 0;
      
    for (i = 0; i < biases.size(); ++i) {
      g_biases += grad_biases[i] * grad_biases[i];
      cos_alpha += grad_biases[i] * grad_biases_old[i];
    }
    for (i = 0; i < sinapsis.size(); ++i) {
      g_sinapsis += grad_sinapsis[i] * grad_sinapsis[i];
      cos_alpha += grad_sinapsis[i] * grad_sinapsis_old[i];
    }

    printf("Total gradient biases = %.8e\n", g_biases);
    printf("Total gradient sinapsis = %.8e\n", g_sinapsis);

    // Update the learning rate
    if (ka > 1 && use_line_minimization) {
      cos_alpha /= sqrt(g_biases + g_sinapsis) * sqrt(g_biases_old + g_sinapsis_old);
      // Regularization
      if (cos_alpha < 0)
	learning_rate = learning_rate * learning_inertia + learning_rate / (1 - cos_alpha) * (1 - learning_inertia);

      printf("Angle between consecutive gradients = %d deg\n", (int) (acos(cos_alpha) * 180 / M_PI));
      printf("New minimization step size = %.8e\n", learning_rate);
    }

    printf("Loss function before the step = %.8e\n", loss);


    if (training_algorithm == "sdes") {
      // Perform the steepest descent
      for (i = 0; i < biases.size(); ++i) {
	biases[i] -= learning_rate * grad_biases[i];
      }
      for (i = 0; i < sinapsis.size(); ++i) {
	sinapsis[i] -= learning_rate * grad_sinapsis[i];
      }
    }
    else if (training_algorithm == "annealing-force") {

      // Control if the beta_direction makes sense
      if (energy_beta_equal) beta_direction = 1. / t_start;
      
      // Generate the displacement
      double y, x;
      for (i = 0; i < biases.size(); ++i) {
	y = rand() / (double) RAND_MAX;
	if (beta_direction == 0 || fabs(grad_biases[i]) < ZERO )
	  x = learning_rate * (2 * y - 1);
	else
	  x = log( exp(- learning_rate* beta_direction * grad_biases[i]) +
		   2 * y * sinh(learning_rate * beta_direction * grad_biases[i]))
	    / ( beta_direction* grad_biases[i]);

	// Update
	old_biases[i] = biases.at(i);
	biases.at(i) -= x;
	if (isnan(x)) {
	  cerr << "ERROR: nan number encountered. Look at gradient, beta and learning rate to see if they make sense:" <<endl;
	  cout << "ERROR: nan number encountered. Look at gradient, beta and learning rate to see if they make sense:" <<endl;
	  cout << "       beta = " << scientific << beta_direction << endl;
	  cout << "       learning_rate = " << scientific << learning_rate << endl;
	  cout << "       y = " << scientific << y;
	  exit(EXIT_FAILURE);
	}
      }
      for (i = 0; i < sinapsis.size(); ++i) {
	y = rand() / (double) RAND_MAX;
	if (beta_direction == 0 || fabs(grad_sinapsis[i]) < ZERO)
	  x = 2 * y - 1;
	else
	  x = log( exp(-  learning_rate * beta_direction * grad_sinapsis[i]) +
		   2 * y * sinh( learning_rate* beta_direction * grad_sinapsis[i]))
	    / ( beta_direction* grad_sinapsis[i]);

	// Update
	old_sinapsis[i] =  sinapsis.at(i);
	sinapsis.at(i) -=  x;
	if (isnan(x)) {
	  cerr << "ERROR: nan number encountered. Look at gradient, beta and learning rate to see if they make sense:" <<endl;
	  cout << "ERROR: nan number encountered. Look at gradient, beta and learning rate to see if they make sense:" <<endl;
	  cout << "       beta = " << scientific << beta_direction << endl;
	  cout << "       learning_rate = " << scientific << learning_rate << endl;
	  exit(EXIT_FAILURE);
	}
      }

      // Perform the new loss calculation
      new_loss = new_loss_forces = 0;
      for (i = 0; i < batch_size; ++i) {
	config_index = (start_config_index +i) % training_set ->get_ndata();

	// Predict features
	PredictFeatures(1, training_set->get_feature(config_index), predictions + get_npredictions()*i);

	// Get forces
	GetForces(predicted_forces +N_nodes.at(0) *i);

	// Compute the loss
	new_loss += loss_function(get_npredictions(), predictions + get_npredictions() *i,
				  training_set->get_values(config_index)) / batch_size;
	new_loss_forces += loss_function(N_nodes.at(0), predicted_forces + N_nodes.at(0) *i,
				     training_set->get_forces(config_index)) / (batch_size * N_nodes.at(0));
      }

      cout << "-> ANNEALING:" << endl;
      printf("-> Loss forces after the step = %.8e\n", new_loss_forces);
      printf("-> Delta Loss = %.8e  (old loss = %.8e)\n", new_loss_forces - loss_forces, loss_forces);
      printf("-> Current temperature = %.8e\n", t_start);
      // Accept criteria
      x = rand() / (double) RAND_MAX;
      double delta_e, delta_f;
      delta_e = new_loss - loss;
      delta_f = new_loss_forces - loss_forces;


      double control = delta_f / t_start - delta_e * (beta_direction - 1. / t_start);
      printf("Extracted %.8lf; control %.8lf;", x, exp(-control));

      if (x > exp( -control) || isnan( exp(-control))) {
	// Reject the move
	for (i = 0; i< biases.size(); ++i)
	  biases.at(i) = old_biases[i];
	for (i = 0; i < sinapsis.size(); ++i)
	  sinapsis.at(i) = old_sinapsis[i];
	
	cout << " rejected" << endl;
      } else {
	accept_rate += 1 / (double) check_accept_rate;
	cout << " accepted" << endl;
      }

      // Update the temperature
      t_start *= (1 - t_decay);
      if (t_start < ZERO) t_start = ZERO;
      
      // Update the accept rate
      if ((ka+1) % check_accept_rate == 0 ) {
	printf("accept rate = %.4lf\n", accept_rate);
	learning_rate *= 1 + (accept_rate - 0.5);
	accept_rate = 0;
      }
    }

    // // Compute all the predictions
    // PredictFeatures(batch_size, training_set->get_feature(start_config_index),
    // 		    predictions);

    // // Compute the loss function
    // printf("Loss function after the step = %.8e\n",
    // 	   loss_function(batch_size * get_npredictions(),
    // 			 predictions,
    // 			 training_set->get_values(start_config_index)) / batch_size);

    // Use the test set to compute the loss


    for (i = 0; i < test_set->get_ndata(); ++i) {
      PredictFeatures(1, test_set->get_feature(i), predictions);

      if (train_with_forces) {
	GetForces(predicted_forces + N_nodes.at(0) * i);
	loss_forces += loss_function(N_nodes.at(0), predicted_forces +  N_nodes.at(0) * i,
				     test_set->get_forces(i)) / (test_set->get_ndata() * N_nodes.at(0));	
      }
      loss += loss_function(get_npredictions(), predictions + get_npredictions() * i,
			    test_set->get_values(i)) / test_set->get_ndata();
    }

    printf("Loss function of the test set = %.8e\n", loss);
    if (train_with_forces)
      printf("Forces Loss function of the test set = %.8e\n", loss_forces);

    // Update the configuration batch
    start_config_index = rand() % training_set->get_ndata();
  }
}

void NeuralNetwork::StepDescent(const double * gradients_loss, double * grad_biases, double * grad_sinapsis,
				double * grad_first_layer) { 
  // Prepare an auxiliary network for the back propagation
  int i, j, k, layer;
  double tmp_deriv;
  
  //vector <double> aux_neurons;
  double * aux_neurons = (double*) calloc(sizeof(double), neurons.size());
  
  // For each outcoming a new derivative must be computed
  // Clear all
  layer = N_hidden_layers;
  int an_index = 0;
  
  for (k = 0; k < N_nodes.at(layer + 1) ; ++k)
    aux_neurons[ get_neuron_index(layer + 1, k) ] = gradients_loss[k];

  
  // // Start from the last hidden layer    
  // for (i = 0; i < N_nodes.at(layer); ++i) {
  //   tmp_deriv = 0;
  //   for (k = 0; k < N_nodes.at(layer + 1) ; ++k) {
  //     tmp_deriv += sinapsis.at( get_sinapsis_index(layer, i, k)) *
  // 	aux_neurons[get_neuron_index(layer + 1, k)];
  //   }
  //   aux_neurons[ get_neuron_index(layer, i)] = tmp_deriv;
  // }

  
  // Proceed with the back propagation (the first layer is not needed)
  // Therefore the activation function is always used
  for (layer = N_hidden_layers ; layer >= 0; --layer) {
    for (i = 0; i < N_nodes.at(layer); ++i) {
      tmp_deriv = 0;
      for (k = 0; k < N_nodes.at(layer+ 1) ; ++k) {
	// If the layer is not the first, then you have to consider the activation function
	if (layer > 0) {
	  tmp_deriv += aux_neurons[get_neuron_index(layer+1, k)] *
	    sinapsis.at( get_sinapsis_index(layer, i, k)) *
	    diff_activation( neurons.at( get_neuron_index(layer, i)));
	} else {
	  // In the first layer neglect the activation function.
	  tmp_deriv += aux_neurons[get_neuron_index(layer+1, k)] *
	    sinapsis.at( get_sinapsis_index(layer, i, k));
	}
      }
      //for ( k=0; k < an_index; ++k) cout << k << ") " << aux_neurons[k]<< endl;
      aux_neurons[get_neuron_index(layer, i)] = tmp_deriv;
    }
  }

  //fflush(stdout);
  
  // Prepare the gradient
  // Now execute the gradient descent
  for (layer = 1; layer <= N_hidden_layers + 1; ++layer) {
    for (j = 0; j < N_nodes.at(layer); ++j) {
      // Descend the biases
      // cout << "Update biases (layer = " << layer << ")" <<endl;
      // cout << "Neuron " << neurons.at(get_neuron_index(layer, j)) << endl;
      
      // cout << "Indices: " << get_biases_index(layer, j) << " / " << biases.size() << endl;

      // Select if the node value is filtered by the activation function
      
      grad_biases[get_biases_index(layer, j)] +=
	aux_neurons[ get_neuron_index(layer, j)] ;

      // Descend the sinapsis
      for (i = 0; i < N_nodes.at(layer - 1); ++i) {
	// cout << "Update sinapsis (layer = " << layer << ")" << endl;
	// cout << " Neuron 1 : " << neurons.at(get_neuron_index(layer, j)) << endl;
	// cout << " Neuron 2 : " << neurons.at(get_neuron_index(layer-1, i)) << endl;
	// cout << "    Total : " << 
	//   diff_activation( neurons.at(get_neuron_index(layer, j))) *
	//   neurons.at(get_neuron_index(layer-1, i)) << endl;

	// If the layer is connected to the features there is no activation function
	if (layer -1 == 0) {
	  grad_sinapsis[get_sinapsis_index(layer-1, i, j)] +=
	    neurons.at(get_neuron_index(layer-1, i)) *
	    aux_neurons[ get_neuron_index(layer, j) ];
	}else {	 
	  grad_sinapsis[get_sinapsis_index(layer-1, i, j)] +=
	    activation(neurons.at(get_neuron_index(layer-1, i))) *
	    aux_neurons[ get_neuron_index(layer, j) ];
	}
      }
    }
  }

  // If the grad_first_layer is not a null pointer, return the gradient respect to the first layer (forces)
  if ( grad_first_layer != NULL) 
    for (i = 0; i < N_nodes.at(0); ++i) 
      grad_first_layer[i] = aux_neurons[get_neuron_index(0, i)];

  // Free memory
  free (aux_neurons);
}



void NeuralNetwork::PredictFeatures(int n_data, const double * features, double * values, bool rescale) {
  // Forward propagation
  int i, j, k, h;
  double neuron_value;

  //cout << "Start feature prediction..." << endl;
  

  // i Sum over the data
  for (i = 0; i < n_data; ++i) {
    //cout << i << ") data" <<endl;

    // Prepare the features on the first layer
    for (k = 0; k < N_nodes.at(0); ++k) {
      neurons[k] = features[N_nodes.at(0) * i + k];

      // Check if the first layer must be rescaled
      if (rescale) {
	neurons[k] = (neurons[k] - input_mean[k]) / input_sigma[k]; 
      }
    }

    
    // Propagate the data over the network
    // j = Sum over the hidden layers and the output
    for ( j = 0; j < N_hidden_layers + 1; ++j) {
      //cout << "  ->" << j << ") layer" << endl;  
      //  h, k = Sum over the neurons between the two layers
      for (k = 0; k < N_nodes.at(j+1); ++k) {
	// Add the neuron bias
	neuron_value = biases.at( get_biases_index(j+1, k));
	
	// Get the sinapsis result
	for (h = 0; h < N_nodes.at(j); ++h) {

	  // Check if the activation function must be applied in the first layer
	  if (j > 0) 
	    neuron_value += sinapsis.at( get_sinapsis_index(j, h, k)) *
	      activation(neurons.at( get_neuron_index(j, h)));
	  else
	    neuron_value += sinapsis.at( get_sinapsis_index(j, h, k)) *
	      neurons.at( get_neuron_index(j, h));	     
	}
	neurons.at(get_neuron_index(j+1, k)) = neuron_value;
	if (j == N_hidden_layers) {
	  values[get_npredictions() * i + k] = neuron_value;
	}
      }
    }
  }

  // Rescale the values to the final value
  if (rescale) 
    for (j = 0; j < n_data; ++j)
      for (i = 0; i < get_npredictions(); ++i) 	
	values[get_npredictions()* j + i] = output_mean[i] + output_sigma[i] * values[get_npredictions()* j + i];

}


// Arctan activation function
double NeuralNetwork::activation(double x) {
  return atan(x);
}

double NeuralNetwork::diff_activation(double x) {
  return 1.0/ ( 1. + x*x);
}


double loss_minsquare(int n, const double * pred, const double * target) {
  int i;
  double loss = 0;
  for (i = 0; i < n; ++i)
    loss += (pred[i] - target[i]) * (pred[i] - target[i]);

  return loss;
}

void gradient_minsquare(int n, const double * pred, const double * target, double * grad) {
  int i;
  for (i = 0; i < n; ++i) {
    grad[i] = 2 * (pred[i] - target[i]) ;
  }
}


void NeuralNetwork::set_learning_step(double s) {learning_rate = s;}
double NeuralNetwork::get_learning_step(void) {return learning_rate;}
void NeuralNetwork::set_max_steps(int s) {n_step_max = s;}
int NeuralNetwork::get_max_steps(void) {return n_step_max;}
void NeuralNetwork::set_learning_inertia(double s) {learning_inertia = s;}
double NeuralNetwork::get_learning_inertia(void) {return learning_inertia;}


// Test the gradient
void NeuralNetwork::TestGradient(Dataset * training_set,
				 double (*loss)(int, const double * , const double *),
				 void (*gradient_loss) (int, const double *, const double *, double *),
				 int biases_index, int sinapsis_index) {
  // Compute the loss and the gradient by variing first the biases index
  int i, j;

  double * gradient_biases = (double*) calloc(sizeof(double), biases.size());
  double * gradient_sinapsis = (double*) calloc(sizeof(double), sinapsis.size());

  double original_biases = biases.at(biases_index);
  double original_sinapsis = sinapsis.at(sinapsis_index);

  double *features = (double*) calloc(sizeof(double), get_npredictions() * training_set->get_ndata());
  double *grad_loss = (double*) calloc(sizeof(double), get_npredictions());
  double loss_value;

  cout << endl << endl;
  cout << " * * * * * * * * * *" << endl;
  cout << " *                 *" << endl;
  cout << " *    TEST GRAD    *" << endl;
  cout << " *                 *" << endl;
  cout << " * * * * * * * * * *" << endl;
  cout << endl << endl;

  cout << " => BIASES " << endl;
  cout << "# step loss gradient" << endl;
  
  for (i = 0; i < n_step_max; ++i) {
    biases.at(biases_index) = original_biases + learning_rate * i;

    // Clean the gradients
    for (j = 0; j < biases.size(); ++j) gradient_biases[j] = 0;
    for (j = 0; j < sinapsis.size(); ++j) gradient_sinapsis[j] = 0;

    // Compute the loss function and the gradient
    for (j = 0; j < training_set->get_ndata(); ++j) {
      PredictFeatures(1, training_set->get_feature(j), features + get_npredictions()*j);
      gradient_loss(get_npredictions(), features + get_npredictions()*j,
		    training_set->get_values(j), grad_loss);
      StepDescent(grad_loss, gradient_biases, gradient_sinapsis); // TODO LOOK HERE
    }
    loss_value = loss(get_npredictions() * training_set->get_ndata(),
		      features, training_set->get_values(0));
    
    // Print the loss and the gradient
    printf("%d\t%.8e\t%.8e %.8e %.8e %.8e\n", i, loss_value, gradient_biases[biases_index],
	   features[0], training_set->get_values(0)[0], grad_loss[0]);
  }

  // Reset the biases to the original value
  biases.at(biases_index) = original_biases;
  
  cout << endl;
  cout << " => SINAPSIS" << endl;
  cout <<  "# step loss gradient" << endl;
  
  for (i = 0; i < n_step_max; ++i) {
    sinapsis.at(sinapsis_index) = original_sinapsis + learning_rate * i;

    // Clean the gradients
    for (j = 0; j < biases.size(); ++j) gradient_biases[j] = 0;
    for (j = 0; j < sinapsis.size(); ++j) gradient_sinapsis[j] = 0;

    // Compute the loss function and the gradient
    for (j = 0; j < training_set->get_ndata(); ++j) {
      PredictFeatures(1, training_set->get_feature(j), features + get_npredictions()*j);

      gradient_loss(get_npredictions(), features + get_npredictions()*j,
		    training_set->get_values(j), grad_loss);

      StepDescent(grad_loss, gradient_biases, gradient_sinapsis);
    }
    loss_value = loss(get_npredictions() * training_set->get_ndata(),
		      features, training_set->get_values(0));
    
    // Print the loss and the gradient
    printf("%d\t%.8e\t%.8e\n", i, loss_value, gradient_sinapsis[sinapsis_index]);
  }


  cout << endl;
  cout << " => FORCES " << endl;
  cout <<  "# step loss gradient" << endl;

  double original_pos = * (training_set->get_feature(0));
  double * forces = (double*)calloc(sizeof(double), training_set->get_nfeatures());
  
  for (i = 0; i < n_step_max; ++i) {
    //biases.at(biases_index) = original_biases + learning_rate * i;
    training_set->get_feature(0)[0] = original_pos + learning_rate *i; 
    
    // Clean the gradients
    for (j = 0; j < biases.size(); ++j) gradient_biases[j] = 0;
    for (j = 0; j < sinapsis.size(); ++j) gradient_sinapsis[j] = 0;
    for (j = 0; j < training_set->get_nfeatures(); ++j) forces[j] = 0;

    // Compute the loss function and the gradient
    for (j = 0; j < training_set->get_ndata(); ++j) {
      PredictFeatures(1, training_set->get_feature(j), features + get_npredictions()*j);
      gradient_loss(get_npredictions(), features + get_npredictions()*j,
		    training_set->get_values(j), grad_loss);
      StepDescent(grad_loss, gradient_biases, gradient_sinapsis,
		  forces); // TODO LOOK HERE
    }
    loss_value = loss(get_npredictions() * training_set->get_ndata(),
		      features, training_set->get_values(0));
    
    // Print the loss and the gradient
    printf("%d\t%.8e\t%.8e\n", i, loss_value, forces[0]);
  }


  // Free the memory
  free(gradient_biases);
  free(gradient_sinapsis);
  free(features);
  free(forces);
  free(grad_loss);
}


int NeuralNetwork::get_nbiases() {return biases.size();}
int NeuralNetwork::get_nsinapsis() {return sinapsis.size();}


void NeuralNetwork::Save(string filename) {
  Config cfg;

  Setting &root = cfg.getRoot();

  root.add("NeuralNetwork", Setting::TypeGroup);
  Setting &NN = root["NeuralNetwork"];

  // Add the info on the layer
  NN.add("n_input", Setting::TypeInt) = N_nodes.at(0);
  NN.add("n_output", Setting::TypeInt) = N_nodes.at(N_hidden_layers+1);
  NN.add("n_hidden", Setting::TypeInt) = N_hidden_layers;

  Setting &n_nodes = NN.add("list_nodes_hidden", Setting::TypeArray);
  for (int i = 0; i < N_hidden_layers; ++i) n_nodes.add(Setting::TypeInt) = N_nodes.at(i+1);


  // Add the info on the biases
  Setting &Sbiases = NN.add("biases", Setting::TypeArray);
  for (int i = 0; i < biases.size(); ++i) 
    Sbiases.add(Setting::TypeFloat) = biases.at(i);

  // Add the info on the sinapsis
  Setting &Ssinapsis = NN.add("sinapsis", Setting::TypeArray);
  for (int i = 0; i < sinapsis.size(); ++i)
    Ssinapsis.add(Setting::TypeFloat) = sinapsis.at(i);

  // Write the sigmas and mean of input/output nodes
  Setting &Sinput_sigma = NN.add("input_sigma", Setting::TypeArray);
  Setting &Sinput_mean = NN.add("input_mean", Setting::TypeArray);
  Setting &Soutput_sigma = NN.add("output_sigma", Setting::TypeArray);
  Setting &Soutput_mean = NN.add("output_mean", Setting::TypeArray);

  for (int i = 0; i < N_nodes.at(0); ++i) {
    Sinput_mean.add(Setting::TypeFloat) = input_mean[i];
    Sinput_sigma.add(Setting::TypeFloat) = input_sigma[i];
  }
  for (int i = 0; i < get_npredictions(); ++i) {
    Soutput_mean.add(Setting::TypeFloat) = output_mean[i];
    Soutput_sigma.add(Setting::TypeFloat) = output_sigma[i];
  }
  

  // Now save the file
  cfg.writeFile(filename.c_str());
}


NeuralNetwork::NeuralNetwork(string filename) {
  Config cfg;
  cfg.readFile(filename.c_str());

  // Free the memory of the current object
  
  Setting &root = cfg.getRoot();
  Setting &NN = root["NeuralNetwork"];

  verbosity_level = 1;
  data_dir = ".";

  N_hidden_layers = NN.lookup("n_hidden");

  N_nodes.push_back(NN.lookup("n_input"));
  const Setting &n_nodes_setting = NN["list_nodes_hidden"];
  
  for (int i = 0; i  < N_hidden_layers; ++i) {
    N_nodes.push_back( n_nodes_setting[i] );
  }

  N_nodes.push_back(NN.lookup("n_output"));

  // Load the biases and sinapsis

  const Setting &Sbiases = NN["biases"];
  for (int i = 0; i < Sbiases.getLength(); ++i) {
    biases.push_back(Sbiases[i]);
  }

  const Setting &Ssinapsis = NN["sinapsis"];
  for (int i = 0; i < Ssinapsis.getLength(); ++i)
    sinapsis.push_back(Ssinapsis[i]);

  // Setup the neurons
  for (int layer = 0; layer < N_hidden_layers + 2; ++layer)
    for (int i = 0; i < N_nodes.at(layer); ++i)
      neurons.push_back(0);


  NN.lookupValue("learning_rate", learning_rate);
  NN.lookupValue("learning_inertia", learning_inertia);
  NN.lookupValue("max_steps", n_step_max);

  // Setup the default algorithm
  beta_direction = -1;
  energy_beta_equal = false;
  accept_rate = 10;
  t_decay = 0;
  t_start = 1;
  set_training_algorithm("sdes");

  // Load the input-output rescaling
  const Setting &Sinput_sigma = NN["input_sigma"];
  const Setting &Sinput_mean = NN["input_mean"];
  const Setting &Soutput_sigma = NN["output_sigma"];
  const Setting &Soutput_mean = NN["output_mean"];

  input_sigma = (double*) malloc(sizeof(double) * N_nodes.at(0));
  input_mean = (double*) malloc(sizeof(double) * N_nodes.at(0));

  output_sigma = (double*) malloc(sizeof(double) * get_npredictions());
  output_mean = (double*) malloc(sizeof(double) * get_npredictions());

  if (Sinput_sigma.getLength() != N_nodes.at(0) ||
      Sinput_mean.getLength() != N_nodes.at(0) ||
      Soutput_sigma.getLength() != get_npredictions() ||
      Soutput_mean.getLength() != get_npredictions()) {
    cerr << "Error: the formmating of input/output rescaling parameters" <<endl;
    cerr << "       does not match the network input/output layer size." << endl;
    cout << "Error: the formmating of input/output rescaling parameters" <<endl;
    cout << "       does not match the network input/output layer size." << endl;
    exit(EXIT_FAILURE);
  }

  // Setup the input mean and sigma
  for (int i = 0; i < N_nodes.at(0); ++i) { 
    input_sigma[i] = Sinput_sigma[i];
    input_mean[i] = Sinput_mean[i];
  }
  for (int i = 0; i < get_npredictions(); ++i) {
    output_mean[i] = Soutput_mean[i];
    output_sigma[i] = Soutput_sigma[i];
  }
}


void NeuralNetwork::GetForces(double * forces, bool rescale) {
  // Allocate dumb variables
  double dumb = -1;
  double * g1 = (double*)malloc(sizeof(double) * biases.size());
  double * g2 = (double*)malloc(sizeof(double) * sinapsis.size());

  // Clear the forces
  for (int i = 0; i < N_nodes.at(0); ++i) forces[i] = 0;

  // Compute the forces
  StepDescent( & dumb, g1, g2, forces);

  // Rescale the force unit if required
  if (rescale) {
    for (int i = 0; i < N_nodes.at(0); ++i)
      forces[i] *= output_sigma[0] / input_sigma[i];
  }

  // Free memory
  free(g1);
  free(g2);
}


 void NeuralNetwork::set_training_algorithm(string alg, double t_dec, double t_0) {
  t_decay = t_dec;
  t_start = t_0;
  check_accept_rate = 10;

  
  // Check if the given algorithm is allowed
  if (alg == "sdes") {
    training_algorithm = alg;
    train_with_forces = false;
    use_line_minimization = true;
  }
  else if (alg == "annealing-force") {
    training_algorithm = alg;
    train_with_forces = true;
    use_line_minimization = false;

  }
  else {
    cerr << "Error, the specified algorithm is not allowed for the network training." << endl;
    cerr << "algorithm = " << alg << endl;
    cerr << "ABORT" << endl;
    exit(EXIT_FAILURE);
  }
}


void NeuralNetwork::set_accept_rate(double s) {accept_rate = s;}
void NeuralNetwork::set_beta_direction(double s) {beta_direction = s; if (s < 0) energy_beta_equal = true; else energy_beta_equal = false;}
void NeuralNetwork::set_tdecay(double s) {t_decay = s;}
void NeuralNetwork::set_tstart(double s) {t_start = s;}
void NeuralNetwork::set_check_accept_rate(int s) {check_accept_rate = s;}


double NeuralNetwork::get_beta_direction(){return beta_direction;}
int NeuralNetwork::get_accept_rate(){return accept_rate;}


// Manipulate the network features
void NeuralNetwork::set_biases_value(int index, double value) {biases.at(index) = value;}
void NeuralNetwork::set_sinapsis_value(int index, double value) {sinapsis.at(index) = value;}

double NeuralNetwork::get_biases_value(int index) {return biases.at(index);}
double NeuralNetwork::get_sinapsis_value(int index) {return sinapsis.at(index);}

void NeuralNetwork::update_biases_value(int index, double value) {biases.at(index) += value;}
void NeuralNetwork::update_sinapsis_value(int index, double value) {sinapsis.at(index) += value;}

