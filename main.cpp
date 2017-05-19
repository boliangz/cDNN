#include <iostream>
#include "Eigen"
#include "lstm.h"
#include "mlp.h"
#include "dropout.h"
#include "loss.h"
#include "bi_lstm_with_char.h"


int main() {
//    Eigen::MatrixXd x(4, 3);
//    x << 1.0, 2.0, 3.0,
//         4.0, 5.0, 6.0,
//         7.0, 8.0, 9.0,
//         1.0, 2.0, 3.0;

//    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 3);

//    int inputSize = 4;

    //
    // LSTM test
    //
//    int lstmHiddenDim = 10;
//
//    LSTMParameters lstmParameters;
//    LSTMCache lstmCache;
//    LSTMDiff lstmDiff;
//
//    lstmInit(inputSize, lstmHiddenDim, lstmParameters);
//
//    lstmForward(x, lstmParameters, lstmCache);
//
//    Eigen::MatrixXd dy = Eigen::MatrixXd::Random(lstmHiddenDim, x.cols());
//    lstmBackward(dy, lstmParameters, lstmCache, lstmDiff);
//
//    lstmGradientCheck(dy, lstmParameters, lstmCache, lstmDiff);

    //
    // MLP test
    //
//    int mlpHiddenDim = 10;
//
//    MLPParameters mlpParameters;
//    MLPCache mlpCache;
//    MLPDiff mlpDiff;
//
//    mlpInit(inputSize, mlpHiddenDim, mlpParameters);
//
//    mlpForward(x, mlpParameters, mlpCache);
//
//    Eigen::MatrixXd dy = Eigen::MatrixXd::Random(mlpHiddenDim, x.cols());
//    mlpBackward(dy, mlpParameters, mlpCache, mlpDiff);
//
//    mlpGradientCheck(dy, mlpParameters, mlpCache, mlpDiff);


    //
    // Dropout test
    //
//    DropoutCache dropoutCache;
//    DropoutDiff dropoutDiff;
//    double p = 0.5;
//
//    dropoutForward(x, p, dropoutCache);
//
//    Eigen::MatrixXd dy = Eigen::MatrixXd::Random(x.rows(), x.cols());
//    dropoutBackward(dy, dropoutCache, dropoutDiff);
//
//    dropoutGradientCheck(dy, dropoutCache, dropoutDiff);

    //
    // cross entropy loss test
    //
//    CrossEntropyCache crossEntropyCache;
//    CrossEntropyDiff crossEntropyDiff;
//
//    Eigen::MatrixXd ref(mlpCache.y.rows(), mlpCache.y.cols());
//    ref.setRandom();
//
//    crossEntropyForward(mlpCache.y, ref, crossEntropyCache);
//
//    crossEntropyBackward(crossEntropyCache, crossEntropyDiff);
//
//    crossEntropyGradientCheck(crossEntropyCache, crossEntropyDiff);

    std::vector<Sequence> trainingData;



    //
    // random initialize training data
    //
    int numSequence = 20;
    int tokenLen = 5;
    int sequenceLen = 10;
    int wordDim = 50;
    int charDim = 25;
    int labelSize = 3;
    for (int i = 0; i < numSequence; i++) {
        Sequence s;
        s.wordEmb = Eigen::MatrixXd::Random(wordDim, sequenceLen);
        for (int j = 0; j < sequenceLen; j++) {
            Eigen::MatrixXd charEmb = Eigen::MatrixXd::Random(charDim, tokenLen);
            s.charEmb.push_back(charEmb);
        }
        s.label = Eigen::MatrixXd::Constant(labelSize, sequenceLen, 1);

        trainingData.push_back(s);
    }

    //
    // train network
    //
    int epoch = 10;
    biLSTMCharRun(trainingData, trainingData, epoch);

    return 0;
}