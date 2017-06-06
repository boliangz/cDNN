//
// Created by Boliang Zhang on 5/11/17.
//

#ifndef CDNN_UTILS_H
#define CDNN_UTILS_H

#include <Eigen/Core>
#include <vector>

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & x);

Eigen::MatrixXd tanh(const Eigen::MatrixXd & x);

Eigen::MatrixXd softmax(const Eigen::MatrixXd & x);

std::vector<Eigen::MatrixXd> dsoftmax(const Eigen::MatrixXd & softmaxX);

Eigen::MatrixXd dsigmoid(const Eigen::MatrixXd & sigmoidX);

Eigen::MatrixXd dtanh(const Eigen::MatrixXd & tanhX);

Eigen::MatrixXd * initializeVariable(int row, int column);

void gradientClip(Eigen::MatrixXd & gradient);

#endif //CDNN_UTILS_H