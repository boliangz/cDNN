//
// Created by Boliang Zhang on 5/11/17.
//

#ifndef CDNN_UTILS_H
#define CDNN_UTILS_H

#include <Eigen>


Eigen::MatrixXd sigmoid(Eigen::MatrixXd& x);

Eigen::MatrixXd tanh(Eigen::MatrixXd& x);

Eigen::MatrixXd softmax(Eigen::MatrixXd & x);

std::vector<Eigen::MatrixXd> dsoftmax(Eigen::MatrixXd& softmaxX);

Eigen::MatrixXd dsigmoid(Eigen::MatrixXd& sigmoidX);

Eigen::MatrixXd dtanh(Eigen::MatrixXd& tanhX);

Eigen::MatrixXd initializeVariable(int row, int column);

void gradientClip(Eigen::MatrixXd & gradient);

void printMatrix(Eigen::MatrixXd m);



#endif //CDNN_UTILS_H