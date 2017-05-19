//
// Created by Boliang Zhang on 5/11/17.
//

#ifndef PARSER_LSTM_H
#define PARSER_LSTM_H

#include <vector>
#include <random>
#include <Eigen>
#include "utils.h"



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


void lstmInit(int inputSize,
              int hiddenDimension,
              LSTMParameters & lstmParameters
              ){
    lstmParameters.Wi = initializeVariable(inputSize + hiddenDimension, hiddenDimension);
    lstmParameters.Wf = initializeVariable(inputSize + hiddenDimension, hiddenDimension);
    lstmParameters.Wc = initializeVariable(inputSize + hiddenDimension, hiddenDimension);
    lstmParameters.Wo = initializeVariable(inputSize + hiddenDimension, hiddenDimension);
    lstmParameters.bi = initializeVariable(hiddenDimension, 1);
    lstmParameters.bf = initializeVariable(hiddenDimension, 1);
    lstmParameters.bc = initializeVariable(hiddenDimension, 1);
    lstmParameters.bo = initializeVariable(hiddenDimension, 1);
}


void lstmForward(const Eigen::MatrixXd x,
                 const LSTMParameters & lstmParameters,
                 LSTMCache & lstmCache
                 ) {
    auto Wi = lstmParameters.Wi;
    auto Wf = lstmParameters.Wf;
    auto Wc = lstmParameters.Wc;
    auto Wo = lstmParameters.Wo;
    auto bi = lstmParameters.bi;
    auto bf = lstmParameters.bf;
    auto bc = lstmParameters.bc;
    auto bo = lstmParameters.bo;

    long sequenceLen = x.cols();
    long hiddenDim = Wi.cols();
    long inputSize = Wi.rows() - hiddenDim;

    Eigen::MatrixXd h_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);
    Eigen::MatrixXd c_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);

    lstmCache.x = x;
    lstmCache.h = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    lstmCache.c = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    lstmCache.hi = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    lstmCache.hf = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    lstmCache.ho = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    lstmCache.hc = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);

    for(int i=0; i < sequenceLen; i++){
        Eigen::MatrixXd z(hiddenDim + inputSize, 1);
        z << h_prev,
             x.col(i);

        Eigen::MatrixXd hf_ = (z.transpose() * Wf).transpose() + bf;
        Eigen::MatrixXd hf = sigmoid(hf_);
        Eigen::MatrixXd hi_ = (z.transpose() * Wi).transpose() + bi;
        Eigen::MatrixXd hi = sigmoid(hi_);
        Eigen::MatrixXd ho_ = (z.transpose() * Wo).transpose() + bo;
        Eigen::MatrixXd ho = sigmoid(ho_);
        Eigen::MatrixXd hc_ = (z.transpose() * Wc).transpose() + bc;
        Eigen::MatrixXd hc = tanh(hc_);
        Eigen::MatrixXd c = hf.array() * c_prev.array() + hi.array() * hc.array();
        Eigen::MatrixXd h = ho.array() * tanh(c).array();

        lstmCache.h.col(i) = h;
        lstmCache.c.col(i) = c;
        lstmCache.hi.col(i) = hi;
        lstmCache.hf.col(i) = hf;
        lstmCache.ho.col(i) = ho;
        lstmCache.hc.col(i) = hc;

        h_prev = h;
        c_prev = c;
    }

}


void lstmBackward(const Eigen::MatrixXd & dy,
                  const LSTMParameters & lstmParameters,
                  const LSTMCache & lstmCache,
                  LSTMDiff & lstmDiff
                  ){
    auto Wi = lstmParameters.Wi;
    auto Wf = lstmParameters.Wf;
    auto Wc = lstmParameters.Wc;
    auto Wo = lstmParameters.Wo;

    long sequenceLen = dy.cols();
    long hiddenDim = Wi.cols();
    long inputSize = Wi.rows() - hiddenDim;

    Eigen::MatrixXd dh_next = Eigen::MatrixXd::Zero(hiddenDim, 1);
    Eigen::MatrixXd dc_next = Eigen::MatrixXd::Zero(hiddenDim, 1);

    // initialize parameter diff to zero.
    lstmDiff.Wi_diff = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    lstmDiff.Wf_diff = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    lstmDiff.Wc_diff = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    lstmDiff.Wo_diff = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    lstmDiff.bi_diff = Eigen::MatrixXd::Zero(hiddenDim, 1);
    lstmDiff.bf_diff = Eigen::MatrixXd::Zero(hiddenDim, 1);
    lstmDiff.bc_diff = Eigen::MatrixXd::Zero(hiddenDim, 1);
    lstmDiff.bo_diff = Eigen::MatrixXd::Zero(hiddenDim, 1);
    lstmDiff.x_diff = Eigen::MatrixXd::Zero(inputSize, sequenceLen);

    for(long t = sequenceLen; t --> 0;) {
        Eigen::MatrixXd x_t = lstmCache.x.col(t);
        Eigen::MatrixXd hi_t = lstmCache.hi.col(t);
        Eigen::MatrixXd hf_t = lstmCache.hf.col(t);
        Eigen::MatrixXd hc_t = lstmCache.hc.col(t);
        Eigen::MatrixXd ho_t = lstmCache.ho.col(t);
        Eigen::MatrixXd h_t = lstmCache.h.col(t);
        Eigen::MatrixXd c_t = lstmCache.c.col(t);
        Eigen::MatrixXd dy_t = dy.col(t);

        Eigen::MatrixXd c_prev;
        Eigen::MatrixXd h_prev;
        if (t == 0){
            c_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);
            h_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);
        }
        else {
            c_prev = lstmCache.c.col(t-1);
            h_prev = lstmCache.h.col(t-1);
        }

        Eigen::MatrixXd z(hiddenDim + inputSize, 1);
        z << h_prev, x_t;

        Eigen::MatrixXd dh = dy_t + dh_next;

        Eigen::MatrixXd tanhCt = tanh(c_t);
        Eigen::MatrixXd dc = ho_t.array() * dtanh(tanhCt).array() * dh.array() + dc_next.array();
        Eigen::MatrixXd dho = tanh(c_t).array() * dh.array();
        Eigen::MatrixXd dho_inner = dsigmoid(ho_t).array() * dho.array();
        Eigen::MatrixXd dhf = c_prev.array() * dc.array();
        Eigen::MatrixXd dhf_inner = dsigmoid(hf_t).array() * dhf.array();
        Eigen::MatrixXd dhi = hc_t.array() * dc.array();
        Eigen::MatrixXd dhi_inner = dsigmoid(hi_t).array() * dhi.array();
        Eigen::MatrixXd dhc = hi_t.array() * dc.array();
        Eigen::MatrixXd dhc_inner = dtanh(hc_t).array() * dhc.array();

        Eigen::MatrixXd dWc = z * dhc_inner.transpose();
        Eigen::MatrixXd dbc = dhc_inner;

        Eigen::MatrixXd dWi = z * dhi_inner.transpose();
        Eigen::MatrixXd dbi = dhi_inner;

        Eigen::MatrixXd dWf = z * dhf_inner.transpose();
        Eigen::MatrixXd dbf = dhf_inner;

        Eigen::MatrixXd dWo = z * dho_inner.transpose();
        Eigen::MatrixXd dbo = dho_inner;

        Eigen::MatrixXd dz = Wi * dhi_inner + Wc * dhc_inner + Wo * dho_inner + Wf * dhf_inner;
        Eigen::MatrixXd dx = dz.block(hiddenDim, 0, inputSize, 1);

        // update dh_next and dc_next
        dh_next = dz.block(0, 0, hiddenDim, 1);
        dc_next = hf_t.array() * dc.array();

        lstmDiff.Wi_diff += dWi;
        lstmDiff.Wf_diff += dWf;
        lstmDiff.Wc_diff += dWc;
        lstmDiff.Wo_diff += dWo;
        lstmDiff.bi_diff += dbi;
        lstmDiff.bf_diff += dbf;
        lstmDiff.bc_diff += dbc;
        lstmDiff.bo_diff += dbo;
        lstmDiff.x_diff.col(t) += dx;
    }
}


void paramGradCheck(Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & paramToCheck,
                    Eigen::MatrixXd & paramGrad,
                    LSTMParameters & lstmParameters,
                    LSTMCache & lstmCache
){
    Eigen::MatrixXd x = lstmCache.x;
    std::cout.precision(15);

    int num_checks = 10;
    float delta = 10e-5;

    assert(paramToCheck.rows() == paramGrad.rows() || paramToCheck.cols() == paramGrad.cols());

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(paramToCheck.rows()));
        int randCol = 0 + (rand() % (int)(paramToCheck.cols()));

        float originalVal = paramToCheck(randRow, randCol);

        LSTMCache gradCheckCache0;

        paramToCheck(randRow, randCol) = originalVal - delta;

        lstmForward(x, lstmParameters, gradCheckCache0);

        LSTMCache gradCheckCache1;

        paramToCheck(randRow, randCol) = originalVal + delta;

        lstmForward(x, lstmParameters, gradCheckCache1);

        paramToCheck(randRow, randCol) = originalVal;

        float analyticGrad = paramGrad(randRow, randCol);

        float numericalGrad = ((gradCheckCache1.h - gradCheckCache0.h).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}

void inputGradCheck(Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & inputGrad,
                    LSTMParameters & lstmParameters,
                    LSTMCache & lstmCache
){
    Eigen::MatrixXd x = lstmCache.x;
    std::cout.precision(15);

    int num_checks = 10;
    float delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(x.rows()));
        int randCol = 0 + (rand() % (int)(x.cols()));

        float originalVal = x(randRow, randCol);

        LSTMCache gradCheckCache0;

        x(randRow, randCol) = originalVal - delta;

        lstmForward(x, lstmParameters, gradCheckCache0);

        LSTMCache gradCheckCache1;

        x(randRow, randCol) = originalVal + delta;

        lstmForward(x, lstmParameters, gradCheckCache1);

        x(randRow, randCol) = originalVal;

        float analyticGrad = inputGrad(randRow, randCol);

        float numericalGrad = ((gradCheckCache1.h - gradCheckCache0.h).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}


void lstmGradientCheck(Eigen::MatrixXd & dy,
                       LSTMParameters & lstmParameters,
                       LSTMCache & lstmCache,
                       LSTMDiff & lstmDiff
                       ){
    std::cout << "=> gradient checking Wi" << std::endl;
    paramGradCheck(dy, lstmParameters.Wi, lstmDiff.Wi_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking Wf" << std::endl;
    paramGradCheck(dy, lstmParameters.Wf, lstmDiff.Wf_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking Wc" << std::endl;
    paramGradCheck(dy, lstmParameters.Wc, lstmDiff.Wc_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking Wo" << std::endl;
    paramGradCheck(dy, lstmParameters.Wo, lstmDiff.Wo_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking bi" << std::endl;
    paramGradCheck(dy, lstmParameters.bi, lstmDiff.bi_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking bf" << std::endl;
    paramGradCheck(dy, lstmParameters.bf, lstmDiff.bf_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking bc" << std::endl;
    paramGradCheck(dy, lstmParameters.bc, lstmDiff.bc_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking bo" << std::endl;
    paramGradCheck(dy, lstmParameters.bo, lstmDiff.bo_diff, lstmParameters, lstmCache);
    std::cout << "=> gradient checking x" << std::endl;
    inputGradCheck(dy, lstmDiff.x_diff, lstmParameters, lstmCache);
}

void lstmParamUpdate(const double learningRate,
                     LSTMParameters & lstmParameters,
                     LSTMDiff & lstmDiff) {
    gradientClip(lstmDiff.Wi_diff);
    lstmParameters.Wi -= learningRate * lstmDiff.Wi_diff;
    gradientClip(lstmDiff.Wf_diff);
    lstmParameters.Wf -= learningRate * lstmDiff.Wf_diff;
    gradientClip(lstmDiff.Wc_diff);
    lstmParameters.Wc -= learningRate * lstmDiff.Wc_diff;
    gradientClip(lstmDiff.Wo_diff);
    lstmParameters.Wo -= learningRate * lstmDiff.Wo_diff;
    gradientClip(lstmDiff.bi_diff);
    lstmParameters.bi -= learningRate * lstmDiff.bi_diff;
    gradientClip(lstmDiff.bf_diff);
    lstmParameters.bf -= learningRate * lstmDiff.bf_diff;
    gradientClip(lstmDiff.bc_diff);
    lstmParameters.bc -= learningRate * lstmDiff.bc_diff;
    gradientClip(lstmDiff.bo_diff);
    lstmParameters.bo -= learningRate * lstmDiff.bo_diff;
}

#endif //PARSER_LSTM_H
