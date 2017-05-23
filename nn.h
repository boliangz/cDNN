//
// Created by Boliang Zhang on 5/19/17.
//

#ifndef CDNN_NN_H
#define CDNN_NN_H

#include <Eigen>
#include "loader.h"

//
// MLP implementation
//
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

void mlpInit(const int inputSize, const int hiddenDimension, MLPParameters & mlpParameters);

void mlpForward(const Eigen::MatrixXd x, const MLPParameters & mlpParameters, MLPCache & mlpCache);

void mlpBackward(const Eigen::MatrixXd & dy, const MLPParameters & mlpParameters, const MLPCache & mlpCache,
                 MLPDiff & mlpDiff);

void paramGradCheck(const Eigen::MatrixXd & dy, Eigen::MatrixXd & paramToCheck, const Eigen::MatrixXd & paramGrad,
                    MLPParameters & mlpParameters, const MLPCache & mlpCache);


void inputGradCheck(const Eigen::MatrixXd & dy, Eigen::MatrixXd & inputGrad,
                    const MLPParameters & mlpParameters, const MLPCache & mlpCache);

void mlpGradientCheck(const Eigen::MatrixXd & dy, MLPParameters & mlpParameters, const MLPCache & mlpCache,
                      const MLPDiff & mlpDiff);

void mlpParamUpdate(double learningRate,
                    MLPParameters & mlpParameters,
                    MLPDiff & mlpDiff);

//
// LSTM implementation
//
struct LSTMCache {
    Eigen::MatrixXd x;
    Eigen::MatrixXd hi;
    Eigen::MatrixXd hf;
    Eigen::MatrixXd ho;
    Eigen::MatrixXd hc;
    Eigen::MatrixXd c;
    Eigen::MatrixXd h;
};

struct LSTMParameters{
    Eigen::MatrixXd Wi;
    Eigen::MatrixXd Wf;
    Eigen::MatrixXd Wc;
    Eigen::MatrixXd Wo;
    Eigen::MatrixXd bi;
    Eigen::MatrixXd bf;
    Eigen::MatrixXd bc;
    Eigen::MatrixXd bo;
};

struct LSTMDiff {
    Eigen::MatrixXd Wi_diff;
    Eigen::MatrixXd Wf_diff;
    Eigen::MatrixXd Wc_diff;
    Eigen::MatrixXd Wo_diff;
    Eigen::MatrixXd bi_diff;
    Eigen::MatrixXd bf_diff;
    Eigen::MatrixXd bc_diff;
    Eigen::MatrixXd bo_diff;
    Eigen::MatrixXd x_diff;

    LSTMDiff& operator+=(LSTMDiff & a) {
        this->Wi_diff += a.Wi_diff;
        this->Wf_diff += a.Wf_diff;
        this->Wc_diff += a.Wc_diff;
        this->Wo_diff += a.Wo_diff;
        this->bi_diff += a.bi_diff;
        this->bf_diff += a.bf_diff;
        this->bc_diff += a.bc_diff;
        this->bo_diff += a.bo_diff;
        return *this;
    }
};

void lstmInit(const int inputSize, const int hiddenDimension, LSTMParameters & lstmParameters);


void lstmForward(const Eigen::MatrixXd x, const LSTMParameters & lstmParameters, LSTMCache & lstmCache);


void lstmBackward(const Eigen::MatrixXd & dy, const LSTMParameters & lstmParameters, const LSTMCache & lstmCache,
                  LSTMDiff & lstmDiff);


void paramGradCheck(const Eigen::MatrixXd & dy, Eigen::MatrixXd & paramToCheck, const Eigen::MatrixXd & paramGrad,
                    LSTMParameters & lstmParameters, const LSTMCache & lstmCache);

void inputGradCheck(const Eigen::MatrixXd & dy, Eigen::MatrixXd & inputGrad,
                    const LSTMParameters & lstmParameters, const LSTMCache & lstmCache);


void lstmGradientCheck(const Eigen::MatrixXd & dy, LSTMParameters & lstmParameters, const LSTMCache & lstmCache,
                       const LSTMDiff & lstmDiff);

void lstmParamUpdate(const double learningRate, LSTMParameters & lstmParameters, LSTMDiff & lstmDiff);

//
// Bi-lstm implementation
//
struct BiLSTMCache {
    LSTMCache fwdLSTMCache;
    LSTMCache bwdLSTMCache;
    Eigen::MatrixXd h;
};

struct BiLSTMParameters{
    LSTMParameters fwdLSTMParameters;
    LSTMParameters bwdLSTMParameters;
};

struct BiLSTMDiff {
    LSTMDiff fwdLSTMDiff;
    LSTMDiff bwdLSTMDiff;
    Eigen::MatrixXd x_diff;
};

void biLSTMInit(const int inputSize, const int hiddenDimension, BiLSTMParameters & biLSTMParameters);


void biLSTMForward(const Eigen::MatrixXd x, const BiLSTMParameters & biLSTMParameters, BiLSTMCache & BiLSTMCache);


void biLSTMBackward(const Eigen::MatrixXd & dy, const BiLSTMParameters & biLSTMParameters, const BiLSTMCache & BiLSTMCache,
                    BiLSTMDiff & BiLSTMDiff);


void paramGradCheck(const Eigen::MatrixXd & dy, Eigen::MatrixXd & paramToCheck, const Eigen::MatrixXd & paramGrad,
                    BiLSTMParameters & biLSTMParameters, const BiLSTMCache & BiLSTMCache);

void inputGradCheck(const Eigen::MatrixXd & dy, Eigen::MatrixXd & inputGrad,
                    const BiLSTMParameters & biLSTMParameters, const BiLSTMCache & BiLSTMCache);


void biLSTMGradientCheck(const Eigen::MatrixXd & dy, BiLSTMParameters & biLSTMParameters, const BiLSTMCache & BiLSTMCache,
                         const BiLSTMDiff & BiLSTMDiff);

void biLSTMParamUpdate(const double learningRate, BiLSTMParameters & biLSTMParameters, BiLSTMDiff & BiLSTMDiff);


//
// Dropout implementation
//
struct DropoutCache {
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
    Eigen::MatrixXd mask;
};

struct DropoutDiff {
    Eigen::MatrixXd x_diff;
};


void dropoutForward(const Eigen::MatrixXd & x, const double dropoutRate, DropoutCache & dropoutCache);


void dropoutBackward(const Eigen::MatrixXd & dy, const DropoutCache & dropoutCache, DropoutDiff & dropoutDiff);

void dropoutInputUpdate(double learningRate,
                        const Sequence & s,
                        Eigen::MatrixXd & wordEmbedding,
                        DropoutDiff & dropoutDiff);

void inputGradCheck(const Eigen::MatrixXd & dy, const DropoutCache & dropoutCache, const Eigen::MatrixXd & inputGrad);


void dropoutGradientCheck(const Eigen::MatrixXd & dy, const DropoutCache & dropoutCache, const DropoutDiff & dropoutDiff);

//
// Crossentropy loss implementation
//
struct CrossEntropyCache {
    Eigen::MatrixXd pred;
    Eigen::MatrixXd ref;
    Eigen::MatrixXd loss;
};

struct CrossEntropyDiff {
    Eigen::MatrixXd pred_diff;
};


void crossEntropyForward(const Eigen::MatrixXd & pred, const Eigen::MatrixXd & ref,
                         CrossEntropyCache & crossEntropyCache);

void crossEntropyBackward(const CrossEntropyCache & crossEntropyCache, CrossEntropyDiff & crossEntropyDiff);

void inputGradientCheck(const CrossEntropyCache & crossEntropyCache, const Eigen::MatrixXd & inputGrad);

void crossEntropyGradientCheck(const CrossEntropyCache & crossEntropyCache, const CrossEntropyDiff & crossEntropyDiff);

#endif //CDNN_NN_H
