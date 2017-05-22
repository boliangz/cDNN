//
// Created by Boliang Zhang on 5/17/17.
//

#ifndef PARSER_BI_LSTM_WITH_CHAR_H
#define PARSER_BI_LSTM_WITH_CHAR_H

#include "loader.h"

//
// run network
//
void biLSTMCharRun(const std::vector<Sequence> & training, const std::vector<Sequence> & eval,
                   Eigen::MatrixXd & wordEmbedding, const Eigen::MatrixXd & charEmbedding);

void networkForward(const Sequence & s, Eigen::MatrixXd & loss);

void networkBackward(const Sequence & s);

void networkParamUpdate();

//
// gradient check
//
void networkGradientCheck(const Sequence & s);

void paramGradCheck(const Sequence s, const Eigen::MatrixXd & paramToCheck,
                    const Eigen::MatrixXd & paramGrad);

void inputGradCheck(const Sequence & s);


#endif //PARSER_BI_LSTM_WITH_CHAR_H
