//
// Created by Boliang Zhang on 5/19/17.
//

#ifndef CDNN_NN_H
#define CDNN_NN_H

#include "loader.h"
#include <Eigen/Core>
#include <iostream>

class Layer {

public:
    void copyParameters(const std::map<std::string, Eigen::MatrixXd*>& parameters);

    virtual void forward(const Eigen::MatrixXd & input) {}

    virtual void backward(const Eigen::MatrixXd & dy){}

    void backward(const std::vector<Eigen::MatrixXd> & dy);

    void forward(const std::vector<Eigen::MatrixXd> & input);

    void update(float learningRate);

    void gradientCheck();

    std::map<std::string, Eigen::MatrixXd*> parameters;
    std::map<std::string, Eigen::MatrixXd> cache;
    std::map<std::string, Eigen::MatrixXd> diff;
    std::vector<std::map<std::string, Eigen::MatrixXd> > batchCache;
    std::vector<std::map<std::string, Eigen::MatrixXd> > batchDiff;
    std::string name;
    bool isBatch;
    Eigen::MatrixXd output;
    Eigen::MatrixXd inputDiff;
    std::vector<Eigen::MatrixXd> batchOutput;
    std::vector<Eigen::MatrixXd> batchInputDiff;

    std::mutex mtx;
};

// MLP implementation
class MLP: public Layer {
public:
    MLP(){}
    MLP(int inputSize, int hiddenDim, std::string name);

    MLP(int inputSize, int hiddenDim, std::string name,
        const std::map<std::string, Eigen::MatrixXd*>& parameters);

    void forward(const Eigen::MatrixXd & input);

    void backward(const Eigen::MatrixXd & dy);
};

// LSTM implementation
class LSTM: public Layer {
public:
    LSTM(){}
    LSTM(int inputSize, int hiddenDim, std::string name);
    LSTM(int inputSize, int hiddenDim, std::string name, bool isBatch);

    LSTM(int inputSize, int hiddenDim, std::string name,
         const std::map<std::string, Eigen::MatrixXd*>& parameters);
    LSTM(int inputSize, int hiddenDim, std::string name, bool isBatch,
         const std::map<std::string, Eigen::MatrixXd*>& parameters);

    void forward(const Eigen::MatrixXd & input);

    void backward(const Eigen::MatrixXd & dy);

};


//
// Bi-lstm implementation
//
class BiLSTM: public Layer {
public:
    BiLSTM(){}
    BiLSTM(int inputSize, int hiddenDim, std::string name);
    BiLSTM(int inputSize, int hiddenDim, std::string name, bool isBatch);

    BiLSTM(int inputSize, int hiddenDim, std::string name,
           const std::map<std::string, Eigen::MatrixXd*>& parameters);
    BiLSTM(int inputSize, int hiddenDim, std::string name, bool isBatch,
           const std::map<std::string, Eigen::MatrixXd*>& parameters);

    void forward(const Eigen::MatrixXd & input);

    void backward(const Eigen::MatrixXd & dy);

    LSTM* fwdLSTM;
    LSTM* bwdLSTM;
};

//
// Dropout implementation
//
class Dropout: public Layer {
public:
    Dropout(){}
    Dropout(float dropoutRate, std::string name);

    void forward(const Eigen::MatrixXd & input);

    void backward(const Eigen::MatrixXd & dy);

    void gradientCheck();

    double dropoutRate;
};

//
// Crossentropy loss implementation
//
class CrossEntropyLoss: public Layer {
public:
    CrossEntropyLoss(){}
    CrossEntropyLoss(std::string name);

    void forward(const Eigen::MatrixXd & pred,
                 const Eigen::MatrixXd & ref);

    void backward();

    void gradientCheck();
};

#endif //CDNN_NN_H
