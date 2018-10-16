/****************************************************************************/
/* ObjectNet - Object Orientated Neural Network Builder                     */
/* Version 1.0                                                              */
/* Written by John Murray                                                   */
/* Copyright (c) Murray Computing 1993 - 2017                               */
/* Released as open source under MIT licence                                */
/****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mdanet.h"

float ETA = 0.1;
float ALPHA = 1E-4;
float TOLERANCE  = 1E-4;
#define MAXITER 100

int neuronid = 0;

/****************************************************************************/
/* Neuron Base Class Constructor                                            */
/****************************************************************************/

neuron::neuron (int n) {
  no_of_inputs = n;
  neuronid++;
  id = neuronid;
}

/****************************************************************************/
/* Neuron Output Virtual Function Base                                      */
/****************************************************************************/

float neuron::output () {
  return(value);
}

/****************************************************************************/
/* Neuron Set Current Value (integer and float versions)                    */
/****************************************************************************/

void neuron::setValue(int inp) {
  value = (float) inp;
}

void neuron::setValue(float inp) {
  value = (float) inp;
}

/****************************************************************************/
/* Neuron Get Current Value                                                 */
/****************************************************************************/

float neuron::getValue() {
  return(value);
}

/****************************************************************************/
/* Neuron Add to Error Function                                             */
/****************************************************************************/

void neuron::addError(float e) {
  error += e;
}

/****************************************************************************/
/* Neuron Set Error Function                                                */
/****************************************************************************/

void neuron::setError(float e) {
  error = e;
}

/****************************************************************************/
/* Neuron Get Current Error Function                                        */
/****************************************************************************/

float neuron::getError() {
  return(0.0);
}

/****************************************************************************/
/* Neuron Transfer (sigmoid) Function                                       */
/****************************************************************************/

float neuron::transfer(float x) {
  return(1 / (1 + exp(-x)));
//  return((x >= 0.01) ? 1.0 : 0.0);
}

/****************************************************************************/
/* Neuron Derivative of Transfer (sigmoid) Function                         */
/****************************************************************************/

float neuron::dtransfer(float s) {      // Input is sigmoid value
  return (s * (1 - s));                 // Derivative is s(x)(1-s(x))
}

/****************************************************************************/
/* Connection Class Default Constructor (randomize)                         */
/****************************************************************************/

connect::connect () {
  float u;
  float s;

  u = (float) rand() / (RAND_MAX / 2) - 1;
  s = ((u < 0.0) ? -1.0 : 1.0);         // Take sign
  weight = s * sqrt(-log(s * u)) * 2;   // Convert to normal distribution
  delta = 0.0;                          // Adjustment initialised to zero
}

/****************************************************************************/
/* Connection Class Constructor (set value)                                 */
/****************************************************************************/

connect::connect (float w) {
  weight = w;
  delta = 0.0;
}

/****************************************************************************/
/* Connection Class Get Current Connection Strength Function                */
/****************************************************************************/

float connect::strength () {
  return(weight);
}

/****************************************************************************/
/* Connection Class Adjust Connection Strength Function                     */
/****************************************************************************/

void connect::adjust (float a) {
  delta = ETA * ((1 - ALPHA) * a + ALPHA * delta);     // Add smoothing
  weight += delta;                      // Add adjsutment to weight
}

/****************************************************************************/
/* Connection Class Set Connection Strength Function                        */
/****************************************************************************/

void connect::setWeight(float f) {
  weight = f;
}

/****************************************************************************/
/* Layer Neuron Constructor                                                 */
/****************************************************************************/

layer_neuron::layer_neuron (int n) :
	      neuron (n) {
  weights = new connect[n];
  strengths = new float[n];
}

/****************************************************************************/
/* Layer Neuron Join Neurons Function                                       */
/****************************************************************************/

void layer_neuron::join(neuron **p) {
  int i;
  prev_layer = p;
  printf("Neuron id = %d\n",id);
  printf("Connected to : ");
  for (i=0;i<no_of_inputs;i++)
    printf("%d ",prev_layer[i]->id);
  printf("\nWeights : ");
  for (i=0;i<no_of_inputs;i++)
    printf("%f ",weights[i].strength());
  printf("\n\n");
}

/****************************************************************************/
/* Layer Neuron Output Function                                             */
/****************************************************************************/

float layer_neuron::output () {
  float sum = 0.0;
  int i;

  for (i=0;i<no_of_inputs;i++) {
    sum += prev_layer[i]->output() * weights[i].strength();
  }

  setValue(transfer(sum));
  return(getValue());
}

/****************************************************************************/
/* Layer Neuron Get Weights Function                                        */
/****************************************************************************/

float *layer_neuron::getWeights() {
  for (int i=0;i<no_of_inputs;i++)
    strengths[i] = weights[i].strength();
  return(strengths);
}

/****************************************************************************/
/* Layer Neuron Set Weights Function                                        */
/****************************************************************************/

void layer_neuron::setWeights(float *f) {
  for (int i=0;i<no_of_inputs;i++)
    weights[i].setWeight(f[i]);
}

/****************************************************************************/
/* Layer Neuron Apply Correction Factor                                     */
/****************************************************************************/

void layer_neuron::correct() {
  error = error * dtransfer(value);     // Error is multiplied by derivative
  for (int i=0;i<no_of_inputs;i++) {    // For each connection to next layer
    prev_layer[i]->addError(error * weights[i].strength()); // Add error
    weights[i].adjust(error * prev_layer[i]->getValue());   // Adjust weight
  }
}

/****************************************************************************/
/* Layer Neuron Get Current Error Function                                  */
/****************************************************************************/

float layer_neuron::getError() {
  return(error);
}

/****************************************************************************/
/* Threshold Neuron Constructor                                             */
/****************************************************************************/

threshold_neuron::threshold_neuron() :
  neuron(0) {
  setValue((float) 1.0);
}

/****************************************************************************/
/* Threshold Neuron Output Function                                         */
/****************************************************************************/

float threshold_neuron::output() {
  return(1.0);
}

/****************************************************************************/
/* Network Container Class Constructor                                      */
/****************************************************************************/

network::network (int in, int out, int hid, int layer) {
  int i;                                // Loop variable

  no_of_inputs = in;                    // No of input neurons
  no_of_outputs = out;                  // No of output neurons
  no_of_hidden = hid;                   // No of hidden neurons
  no_of_layers = layer;                 // No of layers
  results = new float[out];             // Store results
  inputs = new neuron*[no_of_inputs+1]; // Create input layer & threshold
  hidden = new neuron*[no_of_hidden+1]; // Create hidden layer & threshold
  outputs = new neuron*[no_of_outputs]; // Create output layer
  for (i=0;i<no_of_inputs;i++)          // For each input neuron
    inputs[i] = new input_neuron;       // Create new neuron
  inputs[no_of_inputs] = new threshold_neuron;    // Create threshold
  for (i=0;i<no_of_hidden;i++) {        // For each hidden neuron
    hidden[i] = new layer_neuron(no_of_inputs+1); // Create new neuron
    hidden[i]->join(inputs);            // Join to previous layer
  }
  hidden[no_of_hidden] = new threshold_neuron;    // Create threshold
  for (i=0;i<out;i++) {                  // For each output neuron
    outputs[i] = new layer_neuron(no_of_hidden+1);// Create new neuron
    outputs[i]->join(hidden);            // Join to previous layer
  }
}

/****************************************************************************/
/* Network Container Class Destructor                                       */
/****************************************************************************/

network::~network () {
  int i;

  for (i=0;i<no_of_inputs;i++)
    delete inputs[i];
  delete [] inputs;
  for (i=0;i<no_of_hidden;i++)
    delete hidden[i];
  delete [] hidden;
  for (i=0;i<no_of_outputs;i++)
    delete outputs[i];
  delete [] outputs;
}

/****************************************************************************/
/* Network Display Parameters Function                                      */
/****************************************************************************/

void network::displayParam() {
}

/****************************************************************************/
/* Network Set Constants                                                    */
/****************************************************************************/

void network::setConstants(float e,float a,float t) {
  ETA = e;
  ALPHA = a;
  TOLERANCE = t;
}

float *network::test (float *obs) {
  static float testfloat[10];
  int i;

  for (i=0;i<no_of_inputs;i++)
    inputs[i]->setValue(obs[i]);

  for (i=0;i<no_of_outputs;i++)
    testfloat[i] = outputs[i]->output();

  return(testfloat);
}

/****************************************************************************/
/* Network Back Propagation Function                                        */
/****************************************************************************/

int network::backprop(float *p) {
  int no_correct = 0;                   // Number of correct predictions
  float output_error;  
  int i;                 // Output error

  for (i=0;i<no_of_hidden;i++)      // For each hidden neuron
    hidden[i]->setError(0.0);           // Reset the weights
  for (i=0;i<no_of_outputs;i++) {       // For each output neuron
    output_error = p[i] - outputs[i]->getValue();
    no_correct += (fabs(output_error) <= TOLERANCE);
    outputs[i]->setError(output_error);
    outputs[i]->correct();              // Apply correction factor
  }
  for (i=0;i<no_of_hidden;i++)          // For each output neuron
    hidden[i]->correct();
  return(no_correct);                   // Return number correct
}

/****************************************************************************/
/* MNIST Main Function                                                            */
/****************************************************************************/

int main () {
  float inp[1000];
  int i,j,k;
  int train = 0;
  int val = 0;
  int nrecs = 0;
  FILE *binary;
//  char *p;
  network n(784,10,50,3);
  int ncounts = 0;
  float* r;
  float val_error;
  float train_error;
  int rc;
  int max_r;
  int max_i;
  
  nrecs = 70000;

  binary = fopen("mnist.bin","rb");
  if (binary == NULL) {
    printf("Unable to open binary file\n");
    exit(9);
  }

  printf("Building network\n\n");
  
  for (i=0;i<MAXITER && train < nrecs;i++) {
    train = 0;
    val = 0;
    for (j=0;j<nrecs;j++) {
      rc = fread(inp,sizeof(float),794,binary);
      if (rc == 0) {
          printf("Unexpected end of file after %d records in iteration %d\n",j,i+1);
          exit(1);
      }
      r = n.test(inp+10);
      max_r = 0;
      for (k=1;k<10;k++) {
        if (r[k] > r[max_r]) {
          max_r = k;
        }
      }
            
      max_i = 0;
      for (k=1;k<10;k++) {
        if (inp[k] > inp[max_i]) {
          max_i = k;
        }
      }

      if (j % 5 != 0) {
        n.backprop(inp);
        train += (max_r == max_i);
      }
      else {
        val += (max_r == max_i);
      }
    }
    rewind(binary);
    printf("Iteration %d - Total Records %d - Training Accuracy %d (%.1f%%) - Val Accuracy %d (%.1f%%)\n",
	    i+1,nrecs,train,(float) 100*train/(nrecs-14000),val,(float) 100*val/14000);
  }
  fclose(binary);
  printf("\nNo of iterations %d\n",i);
}
