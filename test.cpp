//
// Created by Boliang Zhang on 5/21/17.
//
#include "nn.h"
#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


int main() {
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 3);
    int inputSize = 4;

    Eigen::MatrixXd dy;

    // LSTM test
    int lstmHiddenDim = 10;
    dy = Eigen::MatrixXd::Random(lstmHiddenDim, x.cols());

    LSTM lstm(inputSize, lstmHiddenDim, "lstm");
    lstm.forward(x);
    lstm.backward(dy);
    lstm.gradientCheck();

    // Bi-LSTM test
    int biLSTMHiddenDim = 10;
    dy = Eigen::MatrixXd::Random(biLSTMHiddenDim * 2, x.cols());

    BiLSTM biLSTM(inputSize, biLSTMHiddenDim, "biLSTM");
    biLSTM.forward(x);
    biLSTM.backward(dy);
    biLSTM.gradientCheck();

    // MLP test
    int mlpHiddenDim = 10;
    dy = Eigen::MatrixXd::Random(mlpHiddenDim, x.cols());

    MLP mlp(inputSize, mlpHiddenDim, "mlp");
    mlp.forward(x);
    mlp.backward(dy);
    mlp.gradientCheck();

    // Dropout test
    double p = 0.5;

    Dropout dropout(p, "dropout");
    dropout.forward(x);
    dropout.backward(x);
    dropout.gradientCheck();

    // cross entropy loss test
    Eigen::MatrixXd ref(mlp.cache[mlp.name+"_output"].rows(),
                        mlp.cache[mlp.name+"_output"].cols());
    ref.setRandom();
    CrossEntropyLoss crossEntropyLoss("crossEntropyLoss");
    crossEntropyLoss.forward(mlp.cache[mlp.name+"_output"], ref);
    crossEntropyLoss.backward();
    crossEntropyLoss.gradientCheck();

    return 0;
}

