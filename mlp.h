//
// Created by Boliang Zhang on 5/16/17.
//

#ifndef PARSER_MLP_H
#define PARSER_MLP_H

#include <vector>
#include <random>
#include <Eigen>
#include "utils.h"

struct MLPCache {
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
};

struct MLPParameters{
    Eigen::MatrixXd W;
    Eigen::MatrixXd b;
};

struct MLPDiff {
    Eigen::MatrixXd W_diff;
    Eigen::MatrixXd b_diff;
    Eigen::MatrixXd x_diff;
};

void mlpInit(int inputSize,
             int hiddenDimension,
             MLPParameters & mlpParameters
             ){
    mlpParameters.W = initializeVariable(inputSize, hiddenDimension);
    mlpParameters.b = initializeVariable(hiddenDimension, 1);
}


void mlpForward(Eigen::MatrixXd x,
                MLPParameters & mlpParameters,
                MLPCache & mlpCache
                ) {
    auto W = mlpParameters.W;
    auto b = mlpParameters.b;

    Eigen::MatrixXd h_inner = (x.transpose() * W).transpose().colwise() + b.col(0);

    Eigen::MatrixXd h = softmax(h_inner);

    mlpCache.x = x;
    mlpCache.y = h;
}


void mlpBackward(Eigen::MatrixXd & dy,
                 MLPParameters & mlpParameters,
                 MLPCache & mlpCache,
                 MLPDiff & mlpDiff
                 ){
    Eigen::MatrixXd W = mlpParameters.W;

    Eigen::MatrixXd x = mlpCache.x;
    Eigen::MatrixXd y = mlpCache.y;

    long sequenceLen = dy.cols();
    long hiddenDim = W.cols();
    long inputSize = W.rows();

    std::vector<Eigen::MatrixXd> tmp = dsoftmax(y);

    Eigen::MatrixXd dh(hiddenDim, inputSize);

    for (int i = 0; i < sequenceLen; i++) {
        tmp[i] = tmp[i].array().colwise() * dy.col(i).array();
        dh.col(i) = tmp[i].colwise().sum().transpose();
    }

    mlpDiff.W_diff = Eigen::MatrixXd::Zero(inputSize, hiddenDim);
    mlpDiff.b_diff = Eigen::MatrixXd::Zero(hiddenDim, 1);
    mlpDiff.x_diff = Eigen::MatrixXd::Zero(inputSize, sequenceLen);

    for (int i = 0; i < sequenceLen; i++) {
        Eigen::MatrixXd dW = x.col(i) * dh.col(i).transpose();
        Eigen::MatrixXd db = dh.col(i);
        Eigen::MatrixXd dx = W * dh.col(i);

        mlpDiff.W_diff += dW;
        mlpDiff.b_diff += db;
        mlpDiff.x_diff.col(i) += dx;
    }
}


void paramGradCheck(Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & paramToCheck,
                    Eigen::MatrixXd & paramGrad,
                    MLPParameters & mlpParameters,
                    MLPCache & mlpCache
){
    Eigen::MatrixXd x = mlpCache.x;

    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    assert(paramToCheck.rows() == paramGrad.rows() || paramToCheck.cols() == paramGrad.cols());

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(paramToCheck.rows()));
        int randCol = 0 + (rand() % (int)(paramToCheck.cols()));

        double originalVal = paramToCheck(randRow, randCol);

        MLPCache gradCheckCache0;

        paramToCheck(randRow, randCol) = originalVal - delta;

        mlpForward(x, mlpParameters, gradCheckCache0);

        MLPCache gradCheckCache1;

        paramToCheck(randRow, randCol) = originalVal + delta;

        mlpForward(x, mlpParameters, gradCheckCache1);

        paramToCheck(randRow, randCol) = originalVal;

        double analyticGrad = paramGrad(randRow, randCol);

        double numericalGrad = ((gradCheckCache1.y - gradCheckCache0.y).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}

void inputGradCheck(Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & inputGrad,
                    MLPParameters & mlpParameters,
                    MLPCache & mlpCache
){
    Eigen::MatrixXd x = mlpCache.x;

    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(x.rows()));
        int randCol = 0 + (rand() % (int)(x.cols()));

        double originalVal = x(randRow, randCol);

        MLPCache gradCheckCache0;

        x(randRow, randCol) = originalVal - delta;

        mlpForward(x, mlpParameters, gradCheckCache0);

        MLPCache gradCheckCache1;

        x(randRow, randCol) = originalVal + delta;

        mlpForward(x, mlpParameters, gradCheckCache1);

        x(randRow, randCol) = originalVal;

        double analyticGrad = inputGrad(randRow, randCol);

        double numericalGrad = ((gradCheckCache1.y - gradCheckCache0.y).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}


void mlpGradientCheck(Eigen::MatrixXd & dy,
                       MLPParameters & mlpParameters,
                       MLPCache & mlpCache,
                       MLPDiff & mlpDiff
){
    std::cout << "=> gradient checking W" << std::endl;
    paramGradCheck(dy, mlpParameters.W, mlpDiff.W_diff, mlpParameters, mlpCache);
    std::cout << "=> gradient checking x" << std::endl;
    inputGradCheck(dy, mlpDiff.x_diff, mlpParameters, mlpCache);
}

void mlpParamUpdate(double learningRate, MLPParameters & mlpParameters, MLPDiff & mlpDiff) {
    gradientClip(mlpDiff.W_diff);
    mlpParameters.W -= learningRate * mlpDiff.W_diff;
    gradientClip(mlpDiff.b_diff);
    mlpParameters.b -= learningRate * mlpDiff.b_diff;
}

#endif //PARSER_MLP_H
