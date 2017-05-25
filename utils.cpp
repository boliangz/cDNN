//
// Created by Boliang Zhang on 5/19/17.
//

//
// Created by Boliang Zhang on 5/11/17.
//
#include <Eigen/Core>
#include <random>
#include <iostream>
#include "utils.h"


Eigen::MatrixXd sigmoid(Eigen::MatrixXd& x){
    Eigen::MatrixXd result = 1 / (1 + exp(- x.array()));

    return result;
}

Eigen::MatrixXd tanh(Eigen::MatrixXd& x){
    Eigen::MatrixXd tmp = 2 * x;
    Eigen::MatrixXd result = (2 * sigmoid(tmp)).array() - 1;

    return result;
}

Eigen::MatrixXd softmax(Eigen::MatrixXd & x){
    double max = x.maxCoeff();
    Eigen::MatrixXd result = x;
    for (int j = 0; j < x.cols(); j++)
    {
        double sum = 0;
        for (int i = 0; i < x.rows(); i++)
            sum += std::exp(x(i,j) - max);

        double normalizer = std::log(sum);
        for (int k = 0; k < x.rows(); k++)
            result(k,j) = std::exp(x(k,j) - max - normalizer);
    }
    return result;
}


std::vector<Eigen::MatrixXd> dsoftmax(Eigen::MatrixXd& softmaxX){
    std::vector<Eigen::MatrixXd> result;
    for (int i = 0; i < softmaxX.cols(); i++) {
        Eigen::MatrixXd siMultiSj = softmaxX.col(i) * softmaxX.col(i).transpose();
        Eigen::MatrixXd d = softmaxX.col(i).asDiagonal();
        Eigen::MatrixXd r = d - siMultiSj;
        result.push_back(r);
    }

    return result;
}


Eigen::MatrixXd dsigmoid(Eigen::MatrixXd& sigmoidX){
    Eigen::MatrixXd result = sigmoidX.array() * (1 - sigmoidX.array());

    return result;
}

Eigen::MatrixXd dtanh(Eigen::MatrixXd& tanhX){
    Eigen::MatrixXd result = 1 - pow(tanhX.array(), 2);

    return result;
}


Eigen::MatrixXd initializeVariable(int row, int column){
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1, 1);

    double drange = sqrt(6.0 / (row + column));

    Eigen::MatrixXd m(row, column);
    for (int i=0; i < row; i++){
        for (int j=0; j < column; j++){
            m(i, j) = distribution(generator) * drange;
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

void printMatrix(Eigen::MatrixXd m){
    std::cout << m << "\n" << std::endl;
}
