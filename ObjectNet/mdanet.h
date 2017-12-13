/****************************************************************************/
/* ObjectNet - Object Oriented Neural Network Builder Header File           */
/* Version 1.0                                                              */
/* Written by John Murray                                                   */
/* Copyright (c) Murray Computing 1993                                      */
/* Released as open source under MIT licence                                */
/****************************************************************************/

class neuron {
protected:
  float value;
  float error;
public:
  int id;                               // Neuron ID
  int no_of_inputs;                     // Number of networks
  neuron(int n=1);                      // Constructor
  float getValue();                     // Get current value
  virtual float output();               // Output virtual function
  void setValue(int i);                 // Set Value (int) function
  void setValue(float f);               // Set Value (float) version
  void setError(float e);               // Set error
  void addError(float e);               // Add e to error
  virtual float getError();             // Get error function
  virtual void join(neuron **p) {};     // Join pure virtual function
  virtual void correct() {};            // Correct pure virtual function
  float transfer(float a);              // Transfer function
  float dtransfer(float a);             // Derivative of transfer function
};

class connect {
private:
  float weight;	                        // Weight of connection
  float delta;                          // Previous delta
public:
  connect();                            // Default constructor (random)
  connect(float w);                     // Pre-loaded constructor
  float strength();                     // Connection strength
  void adjust(float a);                 // Weight adjustment
  void setWeight(float f);              // Set the weight to a value
};

class input_neuron : public neuron {
};

class layer_neuron : public neuron {
private:
  neuron **prev_layer;                  // Pointer to previous layer
  connect *weights;                     // Weights for each input
  float *strengths;                     // Store connection strengths
public:
  layer_neuron(int n);                  // Constructor
  void join(neuron **p);                // Join to previous layer
  virtual float output();               // Output virtual function
  float *getWeights();                  // Get weights (for save function)
  void setWeights(float *f);            // Set weights (pre-loaded network)
  virtual void correct();               // Error correction
  virtual float getError();             // Get error function
};

class threshold_neuron : public neuron {
public:
  threshold_neuron();                   // Constructor
  virtual float output();               // Output virtual function
};

class network {
private:
  int no_of_inputs;                     // Number of input neurons
  int no_of_outputs;                    // Number of output neurons
  int no_of_hidden;                     // Number of hidden neurons
  int no_of_layers;                     // Number of layers;
  neuron **inputs;                      // Input layer
  neuron **hidden;                      // Hidden layer(s)
  neuron **outputs;                     // Output layer
  float *results;                       // Store for results
public:
  network(int in,int out, int hid, int layer);    // Constructor
  ~network();                           // Destructor
  void displayParam();                  // Print Network Parameters
  float *test(float *obs);		// Test function
  int backprop(float *e);               // Back propagation
  void setConstants(float e,float a,float t);     //Set constant params
};

