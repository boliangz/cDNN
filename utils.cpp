//
// Created by Boliang Zhang on 5/19/17.
//

//
// Created by Boliang Zhang on 5/11/17.
//
#include <Eigen/Core>
#include <random>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include "utils.h"


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
//        std::cout << softmaxX.col(i).sum() << std::endl;
//        std::cout << std::hexfloat << softmaxX.col(i).sum() << std::endl;

        Eigen::MatrixXd siMultiSj = softmaxX.col(i) * softmaxX.col(i).transpose();
//        std::cout << siMultiSj << std::endl;
//        std::cout << siMultiSj.sum() << std::endl;

        Eigen::MatrixXd d = softmaxX.col(i).asDiagonal();
//        std::cout << d << std::endl;
//        std::cout << d.sum() << std::endl;

        Eigen::MatrixXd r = d - siMultiSj;
//        std::cout << r << std::endl;
//        std::cout << r.sum() << std::endl;
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
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> distribution(-1, 1);

    double drange = sqrt(6.0 / (row + column));

    Eigen::MatrixXd m(row, column);
    for (int i=0; i < row; i++){
        for (int j=0; j < column; j++){
//            m(i, j) = distribution(gen) * drange;
            m(i, j) = 0.5;
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
