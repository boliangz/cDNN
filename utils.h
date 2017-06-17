//
// Created by Boliang Zhang on 5/11/17.
//

#ifndef CDNN_UTILS_H
#define CDNN_UTILS_H

#include <Eigen/Core>
#include <vector>

struct Input {};

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & x);

Eigen::MatrixXd tanh(const Eigen::MatrixXd & x);

Eigen::MatrixXd softmax(const Eigen::MatrixXd & x);

std::vector<Eigen::MatrixXd> dsoftmax(const Eigen::MatrixXd & softmaxX);

Eigen::MatrixXd dsigmoid(const Eigen::MatrixXd & sigmoidX);

Eigen::MatrixXd dtanh(const Eigen::MatrixXd & tanhX);

Eigen::MatrixXd * initializeVariable(int row, int column);

void gradientClip(Eigen::MatrixXd & gradient);

template<typename T>
std::vector<std::vector<T>> splitVector(const std::vector<T>& vec, size_t n)
{
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
}

#endif //CDNN_UTILS_H