//
// Created by Boliang Zhang on 5/19/17.
//
#include "utils.h"
#include <Eigen/Core>
#include <random>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

double matrix_sum(Eigen::MatrixXd x){
    double s = 0;
    for (int i = 0; i < x.cols(); ++i)
        for (int j = 0; j < x.rows(); ++j)
            s += x(j, i);
    return s;
}

Eigen::MatrixXd sigmoid(Eigen::MatrixXd& x){
    Eigen::MatrixXd result = 1 / (1 + exp((- x).array()));

    return result;
}

Eigen::MatrixXd tanh(Eigen::MatrixXd& x){
    Eigen::MatrixXd tmp = 2 * x;
    Eigen::MatrixXd result = (2 * sigmoid(tmp)).array() - 1;

    return result;
}

Eigen::MatrixXd softmax(Eigen::MatrixXd & x){
    Eigen::MatrixXd result = x;

    for (int i = 0; i < x.cols(); i++)
    {
        double max = x.col(i).maxCoeff();

        Eigen::MatrixXd exps = exp(x.col(i).array() - max);

        result.col(i) = exps / exps.sum();

    }

    return result;
}


std::vector<Eigen::MatrixXd> dsoftmax(Eigen::MatrixXd& softmaxX){
    std::vector<Eigen::MatrixXd> result;
    for (int i = 0; i < softmaxX.cols(); i++) {
        std::cout.precision(20);

        Eigen::MatrixXd siMultiSj = softmaxX.col(i) * softmaxX.col(i).transpose();

        Eigen::MatrixXd d = softmaxX.col(i).asDiagonal();

        Eigen::MatrixXd r = (d - siMultiSj);

        result.push_back(r);
    }

    return result;
}


Eigen::MatrixXd dsigmoid(Eigen::MatrixXd& sigmoidX){
    Eigen::MatrixXd result = sigmoidX.array() * (1 - sigmoidX.array());

    return result;
}

Eigen::MatrixXd dtanh(Eigen::MatrixXd& tanhX){
    Eigen::MatrixXd result = 1 - tanhX.array() * tanhX.array();

    return result;
}


Eigen::MatrixXd initializeVariable(int row, int column){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> distribution(-1, 1);

    double drange = sqrt(6.0 / (row + column));

    Eigen::MatrixXd m(row, column);
    for (int i=0; i < row; i++){
        for (int j=0; j < column; j++){
            m(i, j) = distribution(gen) * drange;
        }
    }
    return m;
}

void gradientClip(Eigen::MatrixXd & gradient){
    double clip = 5;
    Eigen::MatrixXd c(gradient.rows(), gradient.cols());
    c.fill(clip);
    gradient = gradient.cwiseMin(c);
    gradient = gradient.cwiseMax(-c);
}