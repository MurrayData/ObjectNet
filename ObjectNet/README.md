# ObjectNet

An object oriented library to create neural networks in C++ written on Valentine's Day in 1993

| File | Description          |
| ------- | ---------------- |
| **mda_mnist.cpp**  | Example code to run against MNIST |
| **mdanet.h** | Header file for ObjectNet |
| **mnist.zip** | Zipped version of MNIST stored as binary 32 floating point file (contains mnist.bin) |

Input file is split every 5th record for validation.

To compile on Linux type:

__g++ mda_mnist.cpp -o mda_mnist -O__

I recommend using the -O (optimise) option, unless you experience difficulties, as it speeds up the execution by about 40%

# Sample Output (10 iterations)

```
Building network

Iteration 1 - Total Records 70000 - Training Accuracy 9557 (17.1%) - Val Accuracy 2336 (16.7%)
Iteration 2 - Total Records 70000 - Training Accuracy 21938 (39.2%) - Val Accuracy 5444 (38.9%)
Iteration 3 - Total Records 70000 - Training Accuracy 31056 (55.5%) - Val Accuracy 7766 (55.5%)
Iteration 4 - Total Records 70000 - Training Accuracy 32217 (57.5%) - Val Accuracy 8056 (57.5%)
Iteration 5 - Total Records 70000 - Training Accuracy 34285 (61.2%) - Val Accuracy 8556 (61.1%)
Iteration 6 - Total Records 70000 - Training Accuracy 37056 (66.2%) - Val Accuracy 9228 (65.9%)
Iteration 7 - Total Records 70000 - Training Accuracy 37359 (66.7%) - Val Accuracy 9302 (66.4%)
Iteration 8 - Total Records 70000 - Training Accuracy 37603 (67.1%) - Val Accuracy 9351 (66.8%)
Iteration 9 - Total Records 70000 - Training Accuracy 37731 (67.4%) - Val Accuracy 9376 (67.0%)
Iteration 10 - Total Records 70000 - Training Accuracy 37781 (67.5%) - Val Accuracy 9401 (67.2%)

No of iterations 10
```
