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
    
    cout << "AND Gate results!!\n\n";
    cout << p-> run({0, 0}) << endl; // close to 0
    cout << p-> run({0, 1}) << endl; // close to 0
    cout << p-> run({1, 0}) << endl; // close to 0
    cout << p-> run({1, 1}) << endl; // close to 1
    // Verified by looking at the results on how close they are to 0 and 1
    
    
    p -> set_weights({15, 15, -10});
    cout << "OR Gate results!!\n\n";
    cout << p-> run({0, 0}) << endl; // close to 0
    cout << p-> run({0, 1}) << endl; // close to 1
    cout << p-> run({1, 0}) << endl; // close to 1
    cout << p-> run({1, 1}) << endl; // close to 1
}

