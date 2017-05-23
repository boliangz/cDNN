//
// Created by Boliang Zhang on 5/21/17.
//
#include <iostream>
#include "Eigen"
#include "nn.h"


int main() {
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 3);

    int inputSize = 4;

    //
    // LSTM test
    //
    int lstmHiddenDim = 10;

    LSTMParameters lstmParameters;
    LSTMCache lstmCache;
    LSTMDiff lstmDiff;

    lstmInit(inputSize, lstmHiddenDim, lstmParameters);

    lstmForward(x, lstmParameters, lstmCache);

    Eigen::MatrixXd dy = Eigen::MatrixXd::Random(lstmHiddenDim, x.cols());
    lstmBackward(dy, lstmParameters, lstmCache, lstmDiff);

    std::cout << "lstm gradient check." << std::endl;
    lstmGradientCheck(dy, lstmParameters, lstmCache, lstmDiff);

    //
    // MLP test
    //
    int mlpHiddenDim = 10;

    MLPParameters mlpParameters;
    MLPCache mlpCache;
    MLPDiff mlpDiff;

    mlpInit(inputSize, mlpHiddenDim, mlpParameters);

    mlpForward(x, mlpParameters, mlpCache);

    dy = Eigen::MatrixXd::Random(mlpHiddenDim, x.cols());
    mlpBackward(dy, mlpParameters, mlpCache, mlpDiff);

    std::cout << "mlp gradient check." << std::endl;
    mlpGradientCheck(dy, mlpParameters, mlpCache, mlpDiff);


    //
    // Dropout test
    //
    DropoutCache dropoutCache;
    DropoutDiff dropoutDiff;
    double p = 0.5;

    dropoutForward(x, p, dropoutCache);

    dy = Eigen::MatrixXd::Random(x.rows(), x.cols());
    dropoutBackward(dy, dropoutCache, dropoutDiff);

    std::cout << "dropout gradient check." << std::endl;
    dropoutGradientCheck(dy, dropoutCache, dropoutDiff);

    //
    // cross entropy loss test
    //
    CrossEntropyCache crossEntropyCache;
    CrossEntropyDiff crossEntropyDiff;

    Eigen::MatrixXd ref(mlpCache.y.rows(), mlpCache.y.cols());
    ref.setRandom();

    crossEntropyForward(mlpCache.y, ref, crossEntropyCache);

    crossEntropyBackward(crossEntropyCache, crossEntropyDiff);

    std::cout << "cross entropy gradient check." << std::endl;
    crossEntropyGradientCheck(crossEntropyCache, crossEntropyDiff);

    return 0;
}

