// Author: premkamal | PK | lovelotus
// Learnt from https://www.linkedin.com/learning/training-neural-networks-in-c-plus-plus

#include <iostream>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>
#include <vector>

using namespace std;

class Perceptron {
    public:
        vector<double> weights;
        double bias;
        Perceptron (int inputs, double bias=1.0);
        double run(vector<double> x);
        void set_weights(vector<double> w_init);
        double sigmoid(double x);
};

class MultiLayerPerceptron {
    public:
        vector<int> layers; // number of neurons per layer
        double eta; // learning grade
        double bias; 
        vector<vector<Perceptron>> network; // to hold the complete perceptron network
        vector<vector<double>> values; // holding outer values of the network
        vector<vector<double>> d; // holds error terms
        
        MultiLayerPerceptron(vector<int> inputs, double bias = 1.0, double eta = 0.5);
        
        void set_weights(vector<vector<vector<double>>> w_init);
        void print_weights();
        
        vector<double> run(vector<double> x); // a layer of inputs fed to the multilayer perceptron
        double bp(vector<double> x, vector<double> y); // backpropagation into weights
};

double frand() {
    return (2.0 * (double) rand() / RAND_MAX) - 1.0;
}

Perceptron::Perceptron(int inputs, double bias) {
    this -> bias = bias;
    weights.resize(inputs + 1);
    generate(weights.begin(), weights.end(), frand);
}

double Perceptron::run(vector<double> x) {
    x.push_back(bias);
    double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
    return sigmoid(sum);
}

void Perceptron::set_weights(vector<double> w_init) {
    weights = w_init;
}

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, double eta) {
    this -> layers = layers;
    this -> bias = bias;
    this -> eta = eta;
    
    for (int i = 0; i < layers.size(); i++) {
        values.push_back(vector<double>(layers[i], 0.0));
        network.push_back(vector<Perceptron>());
        // below initialization was missing in the training material
        d.push_back(vector<double>());
        for (int j = 0; j < layers[i]; j++) {
            d[i].push_back(0.0);
        }
        if (i > 0) { // because there are no neurons in the input layer
            for (int j = 0; j < layers[i]; j++) {
                network[i].push_back(Perceptron(layers[i-1], bias));
            }
        }
    }
}

void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>> w_init) {
    for (int i = 0; i < w_init.size(); i++)
        for (int j = 0; j < w_init[i].size(); j++) {
            // since first layer is empty, we are setting network i+1 layer values
            network[i + 1][j].set_weights(w_init[i][j]); 
        }
}

vector<double> MultiLayerPerceptron::run(vector<double> x) {
    values[0] = x;
    for (int i = 1; i < network.size(); i++)
        for (int j = 0; j < layers[i]; j++)
            values[i][j] = network[i][j].run(values[i-1]);
    return values.back();
}

void MultiLayerPerceptron::print_weights() {
    cout <<network.size();
    cout << endl;
    for (int i = 1; i < network.size(); i++) {
        for (int j = 0; j < layers[i]; j++) {
            cout << "Layer " << i + 1 << "Neuron " << j << ": ";
            for (auto &it: network[i][j].weights)
                cout << it << "  ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}

double MultiLayerPerceptron::bp(vector<double>x, vector<double> y) {
    // Backpropagation step by step
    
    // Step 1: Feed a sample to the network
    vector<double> outputs = run(x);
    
    // cout << "Feeding sample to the network complete" << endl;
    
    // Step 2: Calculate the MSE
    vector<double> error;
    double mse = 0.0;
    for (int i = 0; i < y.size(); i++) {
        error.push_back(y[i] - outputs[i]);
        mse += error[i] * error[i];
    }
    mse /= layers.back();
    
    // Step 3: Calculate the output error terms
    for (int i = 0; i < outputs.size(); i++) {
        d.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]);
    }
    
    // Step 4: Calculate the error term of each unit on each layer
    for (int i = network.size() - 2; i > 0; i--) {
        for (int h = 0; h < network[i].size(); h++) {
            double fwd_error = 0.0;
            for (int k = 0; k < layers[i+1]; k++) {
                fwd_error += network[i+1][k].weights[h] * d[i+1][k];
            }
            d[i][h] = values[i][h] * (1-values[i][h]) * fwd_error;
        }
    }
    
    // Step 5 & 6: Calculate the deltas and update the weights
    for (int i = 1; i < network.size(); i++) {
        for (int j = 0; j < layers[i]; j++) {
            for (int k = 0; k < layers[i-1] + 1; k++) { // inputs 0 to layers + 1 bias weight
                double delta;
                if (k == layers[i - 1])
                    delta = eta * d[i][j] * bias;
                else
                    delta = eta * d[i][j] * values[i -1][k];
                network[i][j].weights[k] += delta;
            }
        }
    }
    
    return mse;
}

void trainForIterations(int iterations) {
    MultiLayerPerceptron mlp =  MultiLayerPerceptron({2, 2, 1});
    cout << "Training neural network as an XOR gate..." << endl;
    
    double mse;
    for (int i = 0; i < iterations; i++) {
        mse = 0.0;
        mse += mlp.bp({0, 0}, {0});
        mse += mlp.bp({0, 1}, {1});
        mse += mlp.bp({1, 0}, {1});
        mse += mlp.bp({1, 1}, {0});
        mse = mse / 4.0;
        if (i % 100 == 0)
            cout << "mse minimized after " << iterations << " iterations to " << mse << endl;
        
    }
    
    cout << "\n\n Trained weights (Compared to hard-coded weights):\n\n";
    mlp.print_weights();
    
    cout << "\nXOR Gate results!!" << endl;
    cout << "0 0 = " << mlp.run({0, 0})[0] << endl; // close to 0
    cout << "0 1 = " << mlp.run({0, 1})[0] << endl; // close to 1
    cout << "1 0 = " << mlp.run({1, 0})[0] << endl; // close to 1
    cout << "1 1 = " << mlp.run({1, 1})[0] << endl; // close to 0


}

int main()
{
    srand(time(NULL));
    rand();
    
    cout << "A Logic Gates Perceptron \n\n";
    
    Perceptron *p = new Perceptron(2);
    
    // Determine weights by picking different w0, w1, bias 
    // and plotting (y = x0 * w0 + x1 * w1 + bias) 
    // to find the logic boundary line
    p -> set_weights({10, 10, -15}); // AND
    
    cout << "AND Gate results!!" << endl;
    cout << "0 0 = " << p-> run({0, 0}) << endl; // close to 0
    cout << "0 1 = " << p-> run({0, 1}) << endl; // close to 0
    cout << "1 0 = " << p-> run({1, 0}) << endl; // close to 0
    cout << "1 1 = " << p-> run({1, 1}) << endl; // close to 1
    // Verified by looking at the results on how close they are to 0 and 1
    
    
    p -> set_weights({15, 15, -10});
    cout << "\nOR Gate results!!" << endl;
    cout << "0 0 = " << p-> run({0, 0}) << endl; // close to 0
    cout << "0 1 = " << p-> run({0, 1}) << endl; // close to 1
    cout << "1 0 = " << p-> run({1, 0}) << endl; // close to 1
    cout << "1 1 = " << p-> run({1, 1}) << endl; // close to 1
    
    MultiLayerPerceptron mlp = MultiLayerPerceptron({2, 2, 1});
    // neutron weights of NAND (= - AND) , OR, AND gates
    mlp.set_weights({{{-10, -10, 15},{15, 15, -10}}, {{10, 10, -15}}});
    
    cout << "\nHardcoded MultiLayer Perceptron weights:" << endl;
    mlp.print_weights();
    
    cout << "\nXOR Gate results!!" << endl;
    cout << "0 0 = " << mlp.run({0, 0})[0] << endl; // close to 0
    cout << "0 1 = " << mlp.run({0, 1})[0] << endl; // close to 1
    cout << "1 0 = " << mlp.run({1, 0})[0] << endl; // close to 1
    cout << "1 1 = " << mlp.run({1, 1})[0] << endl; // close to 0
    
    cout << "\n\nTesting a Trained XOR" << endl;
    trainForIterations(2000);
}

