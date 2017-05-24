//
// Created by Boliang Zhang on 5/17/17.
//

#ifndef CDNN_BI_LSTM_H
#define CDNN_BI_LSTM_H

#include "loader.h"

//
// run network
//
void train(const std::vector<Sequence> & training, const std::vector<Sequence> & eval,
                     std::string modelDir, Eigen::MatrixXd & wordEmbedding, const Eigen::MatrixXd & charEmbedding);

void networkForward(const Sequence & s, Eigen::MatrixXd & loss, Eigen::MatrixXd & pred, bool isTrain);

void networkBackward(const Sequence & s);

void networkParamUpdate(Sequence & s, Eigen::MatrixXd & wordEmbedding);

//
// gradient check
//
void networkGradientCheck(const Sequence & s);

void paramGradCheck(const Sequence s, Eigen::MatrixXd & paramToCheck,
                    const Eigen::MatrixXd & paramGrad);

void inputGradCheck(const Sequence & s);


#endif //CDNN_BI_LSTM_H
