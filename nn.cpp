//
// Created by Boliang Zhang on 5/19/17.
//
#include <vector>
#include <random>
#include <Eigen/Core>
#include <iostream>
#include "nn.h"
#include "utils.h"
#include "loader.h"

//
// MLP implementation
//
void mlpInit(const int inputSize,
             const int hiddenDimension,
             MLPParameters & mlpParameters
){
    mlpParameters.W = initializeVariable(inputSize, hiddenDimension);
    mlpParameters.b = initializeVariable(hiddenDimension, 1);
}


void mlpForward(const Eigen::MatrixXd x,
                const MLPParameters & mlpParameters,
                MLPCache & mlpCache
) {
    Eigen::MatrixXd W = mlpParameters.W;
    Eigen::MatrixXd b = mlpParameters.b;

    Eigen::MatrixXd h_inner = (x.transpose() * W).transpose().colwise() + b.col(0);

    Eigen::MatrixXd h = softmax(h_inner);

    mlpCache.x = x;
    mlpCache.y = h;
}


void mlpBackward(const Eigen::MatrixXd & dy,
                 const MLPParameters & mlpParameters,
                 const MLPCache & mlpCache,
                 MLPDiff & mlpDiff
){
    Eigen::MatrixXd W = mlpParameters.W;

    Eigen::MatrixXd x = mlpCache.x;
    Eigen::MatrixXd y = mlpCache.y;

    long sequenceLen = dy.cols();
    long hiddenDim = W.cols();
    long inputSize = W.rows();

    std::vector<Eigen::MatrixXd> tmp = dsoftmax(y);

    Eigen::MatrixXd dh(hiddenDim, inputSize);

    for (int i = 0; i < sequenceLen; i++) {
        tmp[i] = tmp[i].array().colwise() * dy.col(i).array();
        dh.col(i) = tmp[i].colwise().sum().transpose();
    }

    mlpDiff.W_diff = Eigen::MatrixXd::Zero(inputSize, hiddenDim);
    mlpDiff.b_diff = Eigen::MatrixXd::Zero(hiddenDim, 1);
    mlpDiff.x_diff = Eigen::MatrixXd::Zero(inputSize, sequenceLen);

    for (int i = 0; i < sequenceLen; i++) {
        Eigen::MatrixXd dW = x.col(i) * dh.col(i).transpose();
        Eigen::MatrixXd db = dh.col(i);
        Eigen::MatrixXd dx = W * dh.col(i);

        mlpDiff.W_diff += dW;
        mlpDiff.b_diff += db;
        mlpDiff.x_diff.col(i) += dx;
    }
}


void paramGradCheck(const Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & paramToCheck,
                    const Eigen::MatrixXd & paramGrad,
                    MLPParameters & mlpParameters,
                    const MLPCache & mlpCache
){
    Eigen::MatrixXd x = mlpCache.x;

    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    assert(paramToCheck.rows() == paramGrad.rows() || paramToCheck.cols() == paramGrad.cols());

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(paramToCheck.rows()));
        int randCol = 0 + (rand() % (int)(paramToCheck.cols()));

        double originalVal = paramToCheck(randRow, randCol);

        MLPCache gradCheckCache0;

        paramToCheck(randRow, randCol) = originalVal - delta;

        mlpForward(x, mlpParameters, gradCheckCache0);

        MLPCache gradCheckCache1;

        paramToCheck(randRow, randCol) = originalVal + delta;

        mlpForward(x, mlpParameters, gradCheckCache1);

        paramToCheck(randRow, randCol) = originalVal;

        double analyticGrad = paramGrad(randRow, randCol);

        double numericalGrad = ((gradCheckCache1.y - gradCheckCache0.y).array() * dy.array()).sum() / float(2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}

void inputGradCheck(const Eigen::MatrixXd & dy,
                    const Eigen::MatrixXd & inputGrad,
                    const MLPParameters & mlpParameters,
                    const MLPCache & mlpCache
){
    Eigen::MatrixXd x = mlpCache.x;

    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(x.rows()));
        int randCol = 0 + (rand() % (int)(x.cols()));

        double originalVal = x(randRow, randCol);

        MLPCache gradCheckCache0;

        x(randRow, randCol) = originalVal - delta;

        mlpForward(x, mlpParameters, gradCheckCache0);

        MLPCache gradCheckCache1;

        x(randRow, randCol) = originalVal + delta;

        mlpForward(x, mlpParameters, gradCheckCache1);

        x(randRow, randCol) = originalVal;

        double analyticGrad = inputGrad(randRow, randCol);

        double numericalGrad = ((gradCheckCache1.y - gradCheckCache0.y).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}


void mlpGradientCheck(const Eigen::MatrixXd & dy,
                      MLPParameters & mlpParameters,
                      const MLPCache & mlpCache,
                      const MLPDiff & mlpDiff
){
    std::cout << "=> gradient checking W" << std::endl;
    paramGradCheck(dy, mlpParameters.W, mlpDiff.W_diff, mlpParameters, mlpCache);
    std::cout << "=> gradient checking x" << std::endl;
    inputGradCheck(dy, mlpDiff.x_diff, mlpParameters, mlpCache);
}

void mlpParamUpdate(double learningRate, MLPParameters & mlpParameters, MLPDiff & mlpDiff) {
    gradientClip(mlpDiff.W_diff);
    mlpParameters.W -= learningRate * mlpDiff.W_diff;
    gradientClip(mlpDiff.b_diff);
    mlpParameters.b -= learningRate * mlpDiff.b_diff;
}

//
// LSTM implementation
//
void lstmInit(const int inputSize,
              const int hiddenDimension,
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
    Eigen::MatrixXd Wi = lstmParameters.Wi;
    Eigen::MatrixXd Wf = lstmParameters.Wf;
    Eigen::MatrixXd Wc = lstmParameters.Wc;
    Eigen::MatrixXd Wo = lstmParameters.Wo;
    Eigen::MatrixXd bi = lstmParameters.bi;
    Eigen::MatrixXd bf = lstmParameters.bf;
    Eigen::MatrixXd bc = lstmParameters.bc;
    Eigen::MatrixXd bo = lstmParameters.bo;

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
        z << h_prev, x.col(i);

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
    Eigen::MatrixXd Wi = lstmParameters.Wi;
    Eigen::MatrixXd Wf = lstmParameters.Wf;
    Eigen::MatrixXd Wc = lstmParameters.Wc;
    Eigen::MatrixXd Wo = lstmParameters.Wo;

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


void paramGradCheck(const Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & paramToCheck,
                    const Eigen::MatrixXd & paramGrad,
                    LSTMParameters & lstmParameters,
                    const LSTMCache & lstmCache
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

void inputGradCheck(const Eigen::MatrixXd & dy,
                    const Eigen::MatrixXd & inputGrad,
                    const LSTMParameters & lstmParameters,
                    const LSTMCache & lstmCache
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


void lstmGradientCheck(const Eigen::MatrixXd & dy,
                       LSTMParameters & lstmParameters,
                       const LSTMCache & lstmCache,
                       const LSTMDiff & lstmDiff
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

//
// Bi-LSTM implementation
//

void biLSTMInit(const int inputSize,
                const int hiddenDimension,
                BiLSTMParameters & biLSTMParameters){
    LSTMParameters fwdLSTMParameters;
    LSTMParameters bwdLSTMParameters;
    lstmInit(inputSize, hiddenDimension, fwdLSTMParameters);
    lstmInit(inputSize, hiddenDimension, bwdLSTMParameters);
    biLSTMParameters.fwdLSTMParameters = fwdLSTMParameters;
    biLSTMParameters.bwdLSTMParameters = bwdLSTMParameters;
}


void biLSTMForward(const Eigen::MatrixXd x,
                   const BiLSTMParameters & biLSTMParameters,
                   BiLSTMCache & biLSTMCache){
    // bi-directional word lstm forward
    lstmForward(
            x,
            biLSTMParameters.fwdLSTMParameters,
            biLSTMCache.fwdLSTMCache
    );
    lstmForward(
            x.colwise().reverse(),  // backward lstm by reversing input
            biLSTMParameters.bwdLSTMParameters,
            biLSTMCache.bwdLSTMCache
    );

    biLSTMCache.h = Eigen::MatrixXd(biLSTMCache.fwdLSTMCache.h.rows() * 2, biLSTMCache.fwdLSTMCache.h.cols());
    biLSTMCache.h << biLSTMCache.fwdLSTMCache.h, biLSTMCache.bwdLSTMCache.h.colwise().reverse();
}


void biLSTMBackward(const Eigen::MatrixXd & dy,
                    const BiLSTMParameters & biLSTMParameters,
                    const BiLSTMCache & biLSTMCache,
                    BiLSTMDiff & biLSTMDiff){
    Eigen::MatrixXd fwdDy = dy.topRows(dy.rows()/2);
    lstmBackward(
            fwdDy,
            biLSTMParameters.fwdLSTMParameters,
            biLSTMCache.fwdLSTMCache,
            biLSTMDiff.fwdLSTMDiff
    );

    Eigen::MatrixXd bwdDy = dy.bottomRows(dy.rows()/2).colwise().reverse();
    lstmBackward(
            bwdDy,
            biLSTMParameters.bwdLSTMParameters,
            biLSTMCache.bwdLSTMCache,
            biLSTMDiff.bwdLSTMDiff
    );
    biLSTMDiff.x_diff = biLSTMDiff.fwdLSTMDiff.x_diff + biLSTMDiff.bwdLSTMDiff.x_diff.colwise().reverse();
}


void paramGradCheck(const Eigen::MatrixXd & dy,
                    Eigen::MatrixXd & paramToCheck,
                    const Eigen::MatrixXd & paramGrad,
                    BiLSTMParameters & biLSTMParameters,
                    const BiLSTMCache & biLSTMCache
){
    Eigen::MatrixXd x = biLSTMCache.fwdLSTMCache.x;
    std::cout.precision(15);

    int num_checks = 10;
    float delta = 10e-5;

    assert(paramToCheck.rows() == paramGrad.rows() || paramToCheck.cols() == paramGrad.cols());

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(paramToCheck.rows()));
        int randCol = 0 + (rand() % (int)(paramToCheck.cols()));

        float originalVal = paramToCheck(randRow, randCol);

        BiLSTMCache gradCheckCache0;

        paramToCheck(randRow, randCol) = originalVal - delta;

        biLSTMForward(x, biLSTMParameters, gradCheckCache0);

        BiLSTMCache gradCheckCache1;

        paramToCheck(randRow, randCol) = originalVal + delta;

        biLSTMForward(x, biLSTMParameters, gradCheckCache1);

        paramToCheck(randRow, randCol) = originalVal;

        float analyticGrad = paramGrad(randRow, randCol);

        float numericalGrad = ((gradCheckCache1.h - gradCheckCache0.h).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}


void inputGradCheck(const Eigen::MatrixXd & dy,
                    const Eigen::MatrixXd & inputGrad,
                    const BiLSTMParameters & biLSTMParameters,
                    const BiLSTMCache & biLSTMCache){
    Eigen::MatrixXd x = biLSTMCache.fwdLSTMCache.x;
    std::cout.precision(15);

    int num_checks = 10;
    float delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(x.rows()));
        int randCol = 0 + (rand() % (int)(x.cols()));

        float originalVal = x(randRow, randCol);

        BiLSTMCache gradCheckCache0;

        x(randRow, randCol) = originalVal - delta;

        biLSTMForward(x, biLSTMParameters, gradCheckCache0);

        BiLSTMCache gradCheckCache1;

        x(randRow, randCol) = originalVal + delta;

        biLSTMForward(x, biLSTMParameters, gradCheckCache1);

        x(randRow, randCol) = originalVal;

        float analyticGrad = inputGrad(randRow, randCol);

        float numericalGrad = ((gradCheckCache1.h - gradCheckCache0.h).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}


void biLSTMGradientCheck(const Eigen::MatrixXd & dy,
                         BiLSTMParameters & biLSTMParameters,
                         const BiLSTMCache & biLSTMCache,
                         const BiLSTMDiff & biLSTMDiff) {
    std::cout << "####### checking biLSTMDiff.fwdLSTMDiff ########" << std::endl;
    std::cout << "=> gradient checking Wi" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.Wi, biLSTMDiff.fwdLSTMDiff.Wi_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking Wf" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.Wf, biLSTMDiff.fwdLSTMDiff.Wf_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking Wc" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.Wc, biLSTMDiff.fwdLSTMDiff.Wc_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking Wo" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.Wo, biLSTMDiff.fwdLSTMDiff.Wo_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bi" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.bi, biLSTMDiff.fwdLSTMDiff.bi_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bf" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.bf, biLSTMDiff.fwdLSTMDiff.bf_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bc" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.bc, biLSTMDiff.fwdLSTMDiff.bc_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bo" << std::endl;
    paramGradCheck(dy, biLSTMParameters.fwdLSTMParameters.bo, biLSTMDiff.fwdLSTMDiff.bo_diff, biLSTMParameters, biLSTMCache);

    std::cout << "####### checking biLSTMDiff.bwdLSTMDiff ########" << std::endl;
    std::cout << "=> gradient checking Wi" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.Wi, biLSTMDiff.bwdLSTMDiff.Wi_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking Wf" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.Wf, biLSTMDiff.bwdLSTMDiff.Wf_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking Wc" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.Wc, biLSTMDiff.bwdLSTMDiff.Wc_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking Wo" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.Wo, biLSTMDiff.bwdLSTMDiff.Wo_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bi" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.bi, biLSTMDiff.bwdLSTMDiff.bi_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bf" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.bf, biLSTMDiff.bwdLSTMDiff.bf_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bc" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.bc, biLSTMDiff.bwdLSTMDiff.bc_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking bo" << std::endl;
    paramGradCheck(dy, biLSTMParameters.bwdLSTMParameters.bo, biLSTMDiff.bwdLSTMDiff.bo_diff, biLSTMParameters, biLSTMCache);
    std::cout << "=> gradient checking x" << std::endl;

    std::cout << "=> gradient checking wordEmb" << std::endl;
    inputGradCheck(dy, biLSTMDiff.x_diff, biLSTMParameters, biLSTMCache);
}

void biLSTMParamUpdate(const double learningRate, BiLSTMParameters & biLSTMParameters, BiLSTMDiff & biLSTMDiff) {
    lstmParamUpdate(learningRate, biLSTMParameters.fwdLSTMParameters, biLSTMDiff.fwdLSTMDiff);
    lstmParamUpdate(learningRate, biLSTMParameters.bwdLSTMParameters, biLSTMDiff.bwdLSTMDiff);
}


//
// Dropout implementation
//
void dropoutForward(const Eigen::MatrixXd & x,
                    const double dropoutRate,
                    DropoutCache & dropoutCache
) {
    Eigen::MatrixXd mask;
    if (dropoutRate >= 0) {
        // generate dropout mask
        int xSize = x.size();
        std::vector<double> v0(int(xSize * dropoutRate), 0);
        std::vector<double> v1(xSize - v0.size(), 1);
        v0.insert(v0.end(), v1.begin(), v1.end());
        std::random_shuffle(v0.begin(), v0.end());
        double *v_array = &v0[0];
        mask = Eigen::Map<Eigen::MatrixXd>(v_array, x.rows(), x.cols());
    } else {
        mask = dropoutCache.mask;
    }

    // mask on x
    Eigen::MatrixXd y = x.array() * mask.array();

    dropoutCache.x = x;
    dropoutCache.y = y;
    dropoutCache.mask = mask;
}


void dropoutBackward(const Eigen::MatrixXd & dy,
                     const DropoutCache & dropoutCache,
                     DropoutDiff & dropoutDiff
){
    dropoutDiff.x_diff = dy.array() * dropoutCache.mask.array();
}


void dropoutInputUpdate(double learningRate,
                        const Sequence & s,
                        Eigen::MatrixXd & wordEmbedding,
                        DropoutDiff & dropoutDiff) {
    gradientClip(dropoutDiff.x_diff);
    int wordEmbDim = wordEmbedding.rows();
    for (int i = 0; i < s.seqLen; ++i) {
        wordEmbedding.col(s.wordIndex[i]) -= learningRate * dropoutDiff.x_diff.col(i).topRows(wordEmbDim);
    }
}

void inputGradCheck(const Eigen::MatrixXd & dy,
                    const DropoutCache & dropoutCache,
                    const Eigen::MatrixXd & inputGrad
){
    Eigen::MatrixXd x = dropoutCache.x;

    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(x.rows()));
        int randCol = 0 + (rand() % (int)(x.cols()));

        double originalVal = x(randRow, randCol);

        DropoutCache gradCheckCache0;
        gradCheckCache0.mask = dropoutCache.mask;

        x(randRow, randCol) = originalVal - delta;

        dropoutForward(x, -1, gradCheckCache0);

        DropoutCache gradCheckCache1;
        gradCheckCache1.mask = dropoutCache.mask;

        x(randRow, randCol) = originalVal + delta;

        dropoutForward(x, -1, gradCheckCache1);

        x(randRow, randCol) = originalVal;

        double analyticGrad = inputGrad(randRow, randCol);

        double numericalGrad = ((gradCheckCache1.y - gradCheckCache0.y).array() * dy.array()).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}


void dropoutGradientCheck(const Eigen::MatrixXd & dy,
                          const DropoutCache & dropoutCache,
                          const DropoutDiff & dropoutDiff
){
    std::cout << "=> gradient checking x" << std::endl;
    inputGradCheck(dy, dropoutCache, dropoutDiff.x_diff);
}

//
// Crossentropy loss implementation
//
void crossEntropyForward(const Eigen::MatrixXd & pred,
                         const Eigen::MatrixXd & ref,
                         CrossEntropyCache & crossEntropyCache
){
    Eigen::MatrixXd loss = - ref.array() * pred.array().log();

    crossEntropyCache.pred = pred;
    crossEntropyCache.ref = ref;
    crossEntropyCache.loss = loss;
}


void crossEntropyBackward(const CrossEntropyCache & crossEntropyCache,
                          CrossEntropyDiff & crossEntropyDiff
){
    Eigen::MatrixXd pred = crossEntropyCache.pred;
    Eigen::MatrixXd ref = crossEntropyCache.ref;

    Eigen::MatrixXd dpred = - ref.array() / pred.array();

    crossEntropyDiff.pred_diff = dpred;
}

void inputGradientCheck(const CrossEntropyCache & crossEntropyCache,
                        const Eigen::MatrixXd & inputGrad){
    Eigen::MatrixXd pred = crossEntropyCache.pred;
    Eigen::MatrixXd ref = crossEntropyCache.ref;

    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(pred.rows()));
        int randCol = 0 + (rand() % (int)(pred.cols()));

        double originalVal = pred(randRow, randCol);

        CrossEntropyCache gradCheckCache0;

        pred(randRow, randCol) = originalVal - delta;

        crossEntropyForward(pred, ref, gradCheckCache0);

        CrossEntropyCache gradCheckCache1;

        pred(randRow, randCol) = originalVal + delta;

        crossEntropyForward(pred, ref, gradCheckCache1);

        pred(randRow, randCol) = originalVal;

        double analyticGrad = inputGrad(randRow, randCol);

        double numericalGrad = (gradCheckCache1.loss - gradCheckCache0.loss).sum() / (2 * delta);

        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }

}

void crossEntropyGradientCheck(const CrossEntropyCache & crossEntropyCache,
                               const CrossEntropyDiff & crossEntropyDiff
){
    std::cout << "=> gradient checking pred" << std::endl;
    inputGradientCheck(crossEntropyCache, crossEntropyDiff.pred_diff);
}