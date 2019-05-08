/*
 * THE DEEP NEURAL NETWORK CLASS
 */
#ifndef NN_CLASS
#define NN_CLASS

#include <vector>
#include <iostream>

using namespace std;

class Dataset {
  /*
   * The dataset class contains all the information about the features
   * and the expected values. They are used to train and test the
   * networks. Both the features and the expected values are a list of doubles.
   */

private:
  int N_features;
  int N_values;

public:
  int N_data;
  double * features;
  double * forces;
  double * values;
  bool has_forces;
  int verbosity_level;
  string data_dir; // The directory for temporaney input output file during training

  // Constructor
  /*
   * N_data : the number of data in the dataset
   * N_features : for each data, the number of features.
   * N_values : for each data, the number of values.
   *
   * If avabile, the pointers to the data can be passed. In this case, memory wuold not
   * be allocated.
   */
  
  Dataset(int N_data, int N_features, int N_values, double * features = NULL, double * values = NULL);
  ~Dataset();
  
  // Return the number of data in the dataset
  int get_ndata(void);
  int get_nfeatures(void);
  int get_nvalues(void);


  /*
   * Return the features corresponding to the given index.
   */
  double * get_feature(int index = 0);

  /*
   * Return the values corresponding to the given index.
   */
  double * get_values(int index = 0);

  // Get the forces for the corresponding configuration
  double * get_forces(int index = 0);
  
  /*
   * Perform the bootstrap on the dataset.
   * A new Dataset is generated (the old is lost).
   * The data are shuffled randomly with repetitions up to the given new dimension
   * Usually it is done with greater dimension to use the stochastic gradient descent algorithm
   * to train the network with the bootstrap technique.
   *
   * Pay attention: This cancel all the previous values and features accessed via get_freature or
   * get_values
   */
  void bootstrap(int new_dimension);
};


class NeuralNetwork {
private:
  // Number of hidden layers
  int N_hidden_layers;
  
  // Number of notes for each layer
  // The first and the last are, respectively, the incoming and outcoming layers
  vector<int> N_nodes;

  // Biases and sinapsis for the network
  vector<double> biases;
  vector<double> sinapsis;
  vector<double> neurons;

  double learning_rate;
  double learning_inertia;
  int n_step_max;
  int verbosity_level;
  string data_dir;

  bool is_trained;

  // These variables are used by the training algorithm, and setted by the training algorithm
  string training_algorithm;
  bool train_with_forces;
  bool use_line_minimization;
  double t_decay, t_start;
  double beta_direction; // This is the fake temperature for the analytical annealing gradient of the energy
  double accept_rate;
  int check_accept_rate; // Every how many steps do you want to accept the rate?
  bool energy_beta_equal; // If true the beta for energy and forces are the same (annealing)

  // The standard deviation rescaling to be applied in the input and output data
  double * input_sigma, *input_mean;
  double * output_sigma, *output_mean;
  

  

  //int get_inverse_neuron_index(int layer, int index);

  /*
   * The activation function and its derivative.
   * For this particular neural network it is the atan function as defined in math.h
   */
  double activation(double x);
  double diff_activation(double x);
  
public:
  /*
   * The constructor. In the moment the network is built it needs to know
   * how many layers and nodes it has.
   * The number of odes per hidden layer (or a list, or a fixed number for all).
   */
  
  NeuralNetwork(int n_incoming, int n_outcoming, int n_hidden_layer = 0,
		int * nodes_per_hidden_layer = NULL, int nodes_hl = 0);
  // The following constructure just load the network from file
  NeuralNetwork(string filename);
  ~NeuralNetwork();

  /*
   * The training function:
   * 1) training_set : the set of 'features' : 'values' used in the training.
   * 2) test_set : an independent set used to test the convergence against overfitting.
   * 3) batch_size : batch used for the stochastic gradient technique.
   *
   * 4) Loss function (e.g. loss_minsquare)
   * 5) Gradient of the loss function (respect to the targets) (e.g. gradient_minsquare)
   */
  void TrainNetwork(Dataset * training_set, Dataset * test_set, int batch_size,
		    double (*loss_function) (int, const double * , const double *),
		    void (*gradient_loss) (int, const double *, const double *, double *));
  /*
   * Perform the backpropagation algorithm. It is used to compute the
   * network derivative respect to both the parameters and the
   * input layer. It is used to compute forces.
   */
  void StepDescent(const double * gradient_loss, double * grad_biases, double * grad_sinapsis, double * grad_first_layer = NULL);

  /*
   * Compute the forces.
   * The rescaling must be setted to true if you want to predict forces in the same units as the training set. Otherwise
   * the NN normalized units are used.
   */
  void GetForces(double * forces, bool rescale = false);

  /*
   * The following operation process the n_data features into the network
   * and returns the value for each data in the features.
   * Note the values array should be initialized
   * if rescale is true, the rescaling procedure is used so that the output will match the input units
   * otherwise, normized units will be used (useful to stabilize the training procedure)
   */
  void PredictFeatures(int n_data, const double * features, double * values, bool rescale = false);
  
  /*
   * The following function tests if the gradient is correctly computed.
   * By comparing it with the finite difference.
   * The gradient tested is biases and sinapsis (the two index given).
   * learning step is used as step and n_step_max for the number of step.
   */
  void TestGradient(Dataset * training_set, double (*loss) (int, const double * , const double *),
		    void (*gradient_loss) (int, const double *, const double *, double *),
		    int biases_index, int sinapsis_index);

  // Set the learning step
  void set_learning_step(double step);

  // Get the learning step
  double get_learning_step(void);

  // Setup and get the line minmization inertia
  void set_learning_inertia( double );
  double get_learning_inertia( void );

  // Set the max number of steps for the training
  void set_max_steps(int n_max);

  // Get the max number of steps
  int get_max_steps(void);
  

  // Return the number of neurons in the last layer
  int get_npredictions();

  // Get the total number of biases and sinapsis in the network
  int get_nbiases();
  int get_nsinapsis();
  void set_accept_rate(double );
  void set_beta_direction(double);
  void set_tdecay(double);
  void set_tstart(double);
  void set_check_accept_rate(int);

  double get_beta_direction();
  int get_accept_rate();

  // This sets and loads the sinapsis/biases value by the positional index.
  void set_biases_value(int index, double value);
  void set_sinapsis_value(int index, double value);  
  double get_biases_value(int index);
  double get_sinapsis_value(int index);  
  void update_biases_value(int index, double value); // Sum the value to the current value of the biases
  void update_sinapsis_value(int index, double value);  


  // Find the index of a given sinapsis
  int get_neuron_index(int layer, int index);
  int get_sinapsis_index(int starting_layer, int starting_index, int end_index);
  int get_biases_index(int layer, int index);

  // Save the network to a file for a successive loading
  void Save(string filename);

  // Load the network from file
  void Load(string filename);

  // Select the minimization algorithm
  /*
   * Here a list of allowed algorithm:
   * "sdes" : (default)
   *    - This algorithm is a gradient descent with backpropagation
   * "annealing-force" 
   *    - This is the best algorithm for combined energy and forces minimization. The
   *      annealing is operated on the force loss function, while the backpropagation
   *      of the energy is used as an annealing.
   *      In this case the variable t_decay and t_start must be declared. 
   *      It indicates the decay temperature rate for the annealing (exponential) and the initial temperature.
   */
  void set_training_algorithm(string alg, double t_decay = 1e-3, double t_start = 300);
};


// The loss function min square
double loss_minsquare(int n, const double * predictions, const double * targets);
void gradient_minsquare(int n, const double * predictions, const double * targets, double * gradient);

#endif
