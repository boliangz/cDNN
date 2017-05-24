//
// Created by Boliang Zhang on 5/19/17.
//
#include <iostream>
#include <numeric>
#include <ctime>
#include "nn.h"
#include "bi_lstm.h"
#include "loader.h"

//
// network configuration
//
int wordDim = 50;
int wordLSTMHiddenDim = 100;
double learningRate = 0.01;
double dropoutRate = 0.5;

//
// define network parameter, cache and diff
//
BiLSTMParameters biLSTMParameters;
MLPParameters mlpParameters;

BiLSTMCache biLSTMCache;
MLPCache mlpCache;
DropoutCache dropoutCache;
CrossEntropyCache crossEntropyCache;

BiLSTMDiff biLSTMDiff;
MLPDiff mlpDiff;
DropoutDiff dropoutDiff;
CrossEntropyDiff crossEntropyDiff;

void networkForward(const Sequence & s,
                    Eigen::MatrixXd & loss,
                    Eigen::MatrixXd & pred,
                    bool isTrain  // flag used to turn off dropout layer during test
){
    // dropout forward
    if (isTrain)
        dropoutForward(s.wordEmb, dropoutRate, dropoutCache);
    else
        dropoutForward(s.wordEmb, 0, dropoutCache);

    // bi-directional word lstm forward
    biLSTMForward(dropoutCache.y, biLSTMParameters, biLSTMCache);

    // mlp forward
    mlpForward(biLSTMCache.h, mlpParameters, mlpCache);

    // cross entropy forward
    crossEntropyForward(mlpCache.y, s.labelOneHot, crossEntropyCache);

    loss = crossEntropyCache.loss;
    pred = crossEntropyCache.pred;
}

void networkBackward(const Sequence & s){
    // cross entropy backward
    crossEntropyBackward(crossEntropyCache, crossEntropyDiff);

    // mlp backward
    mlpBackward(crossEntropyDiff.pred_diff, mlpParameters, mlpCache, mlpDiff);

    // word lstm backward
    biLSTMBackward(mlpDiff.x_diff, biLSTMParameters, biLSTMCache, biLSTMDiff);

    // dropout backward
    dropoutBackward(biLSTMDiff.x_diff, dropoutCache, dropoutDiff);
}

void networkParamUpdate(Sequence & s,
                        Eigen::MatrixXd & wordEmbedding) {
    mlpParamUpdate(learningRate, mlpParameters, mlpDiff);
    biLSTMParamUpdate(learningRate, biLSTMParameters, biLSTMDiff);
    dropoutInputUpdate(learningRate, s, wordEmbedding, dropoutDiff);
}

void paramGradCheck(const Sequence s,
                    Eigen::MatrixXd & paramToCheck,
                    const Eigen::MatrixXd & paramGrad
){
    dropoutRate = -1;
    std::cout.precision(15);

    int num_checks = 10;
    double delta = 10e-5;

    assert(paramToCheck.rows() == paramGrad.rows() || paramToCheck.cols() == paramGrad.cols());

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(paramToCheck.rows()));
        int randCol = 0 + (rand() % (int)(paramToCheck.cols()));

        double originalVal = paramToCheck(randRow, randCol);

        Eigen::MatrixXd _;

        Eigen::MatrixXd loss0;
        paramToCheck(randRow, randCol) = originalVal - delta;
        networkForward(s, loss0, _, true);

        Eigen::MatrixXd loss1;
        paramToCheck(randRow, randCol) = originalVal + delta;
        networkForward(s, loss1, _, true);

        paramToCheck(randRow, randCol) = originalVal;

        double analyticGrad = paramGrad(randRow, randCol);
        double numericalGrad = (loss1 - loss0).sum() / (2.0 * delta);
        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
    dropoutRate = 0.5;
}

void inputGradCheck(const Sequence & s){
    Sequence sCopy = s;

    Eigen::MatrixXd dWordEmb = dropoutDiff.x_diff.topRows(wordDim);
    std::cout.precision(15);
    dropoutRate = -1;

    int num_checks = 10;
    double delta = 10e-5;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(sCopy.wordEmb.rows()));
        int randCol = 0 + (rand() % (int)(sCopy.wordEmb.cols()));

        double originalVal = sCopy.wordEmb(randRow, randCol);

        Eigen::MatrixXd _;

        Eigen::MatrixXd loss0;
        sCopy.wordEmb(randRow, randCol) = originalVal - delta;
        networkForward(sCopy, loss0, _, true);

        Eigen::MatrixXd loss1;
        sCopy.wordEmb(randRow, randCol) = originalVal + delta;
        networkForward(sCopy, loss1, _, true);

        sCopy.wordEmb(randRow, randCol) = originalVal;

        double analyticGrad = dWordEmb(randRow, randCol);
        double numericalGrad = (loss1 - loss0).sum() / (2 * delta);
        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
    dropoutRate = 0.5;
}

void networkGradientCheck(const Sequence & s){
    std::cout << "####### checking mlpParameters ########" << std::endl;
    std::cout << "=> gradient checking W" << std::endl;
    paramGradCheck(s, mlpParameters.W, mlpDiff.W_diff);
    std::cout << "=> gradient checking b" << std::endl;
    paramGradCheck(s, mlpParameters.W, mlpDiff.W_diff);

    std::cout << "####### checking biLSTMDiff.fwdLSTMDiff ########" << std::endl;
    std::cout << "=> gradient checking Wi" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.Wi, biLSTMDiff.fwdLSTMDiff.Wi_diff);
    std::cout << "=> gradient checking Wf" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.Wf, biLSTMDiff.fwdLSTMDiff.Wf_diff);
    std::cout << "=> gradient checking Wc" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.Wc, biLSTMDiff.fwdLSTMDiff.Wc_diff);
    std::cout << "=> gradient checking Wo" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.Wo, biLSTMDiff.fwdLSTMDiff.Wo_diff);
    std::cout << "=> gradient checking bi" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.bi, biLSTMDiff.fwdLSTMDiff.bi_diff);
    std::cout << "=> gradient checking bf" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.bf, biLSTMDiff.fwdLSTMDiff.bf_diff);
    std::cout << "=> gradient checking bc" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.bc, biLSTMDiff.fwdLSTMDiff.bc_diff);
    std::cout << "=> gradient checking bo" << std::endl;
    paramGradCheck(s, biLSTMParameters.fwdLSTMParameters.bo, biLSTMDiff.fwdLSTMDiff.bo_diff);

    std::cout << "####### checking biLSTMDiff.bwdLSTMDiff ########" << std::endl;
    std::cout << "=> gradient checking Wi" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.Wi, biLSTMDiff.bwdLSTMDiff.Wi_diff);
    std::cout << "=> gradient checking Wf" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.Wf, biLSTMDiff.bwdLSTMDiff.Wf_diff);
    std::cout << "=> gradient checking Wc" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.Wc, biLSTMDiff.bwdLSTMDiff.Wc_diff);
    std::cout << "=> gradient checking Wo" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.Wo, biLSTMDiff.bwdLSTMDiff.Wo_diff);
    std::cout << "=> gradient checking bi" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.bi, biLSTMDiff.bwdLSTMDiff.bi_diff);
    std::cout << "=> gradient checking bf" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.bf, biLSTMDiff.bwdLSTMDiff.bf_diff);
    std::cout << "=> gradient checking bc" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.bc, biLSTMDiff.bwdLSTMDiff.bc_diff);
    std::cout << "=> gradient checking bo" << std::endl;
    paramGradCheck(s, biLSTMParameters.bwdLSTMParameters.bo, biLSTMDiff.bwdLSTMDiff.bo_diff);
    std::cout << "=> gradient checking x" << std::endl;

    std::cout << "=> gradient checking wordEmb" << std::endl;
    inputGradCheck(s);

}

void train(const std::vector<Sequence>& training,
           const std::vector<Sequence>& eval,
           std::string modelDir,
           Eigen::MatrixXd & wordEmbedding,
           const Eigen::MatrixXd & charEmbedding) {
    //
    // initialize parameters
    //
    biLSTMInit(wordDim, wordLSTMHiddenDim, biLSTMParameters);
    int labelSize = training[0].labelOneHot.rows();
    mlpInit(2 * wordLSTMHiddenDim, labelSize, mlpParameters);

    int epoch = 100;
    float bestAcc = 0;
    for (int i = 0; i < epoch; i++) {
        time_t startTime = time(0);
        //
        // train
        //
        std::cout << "=> " << i << " epoch training starts..." << std::endl;

        int numSeqToReport = 1000;
        std::vector<float> epoch_loss;
        std::vector<int> index(training.size());
        std::iota(index.begin(), index.end(), 1);
        std::random_shuffle(index.begin(), index.end());
        for (int j = 0; j < training.size(); ++j){
            Sequence s = training[index[j]-1];

            processData(s, wordEmbedding, charEmbedding);
            //
            // network forward
            //
            Eigen::MatrixXd loss;
            Eigen::MatrixXd pred;

            networkForward(s, loss, pred, true);
            epoch_loss.push_back(loss.sum() / s.seqLen);

            //
            // network backward
            //
            networkBackward(s);

            //
            // network gradient check
            //
//            networkGradientCheck(s);

            //
            // network parameters update
            //
            networkParamUpdate(s, wordEmbedding);

            if ((j + 1) % numSeqToReport == 0){
                std::vector<float> v(epoch_loss.end() - numSeqToReport, epoch_loss.end());
                float average = std::accumulate( v.begin(), v.end(), 0.0)/v.size();
                std::cout << j + 1 << " sequences loss: " << average << std::endl;
            }
        }
        float average = std::accumulate( epoch_loss.begin(), epoch_loss.end(), 0.0)/epoch_loss.size();
        std::cout << "epoch loss: " << average << std::endl;
        time_t trainTime = time(0) - startTime;
        std::printf("time elapsed: %d seconds (%.4f sec/sentence)\n", int(trainTime), float(trainTime) / training.size());

        //
        // eval
        //
        int numTags = 0;
        int numCorrectTags = 0;
        for (int j = 0; j < eval.size(); ++j ) {
            Sequence s = eval[j];
            processData(s, wordEmbedding, charEmbedding);

            Eigen::MatrixXd loss;
            Eigen::MatrixXd pred;
            networkForward(s, loss, pred, false);

            std::vector<int> predLabelIndex;
            Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
            for (int k = 0; k < s.seqLen; ++k) {
                for (int l = 0; l < pred.rows(); ++l) {
                    if (pred(l, k) == maxProba(k, 0))
                        predLabelIndex.push_back(l);
                }
            }
            numTags += s.seqLen;
            for (int k = 0; k < s.seqLen; ++k) {
                if (predLabelIndex[k] == s.labelIndex[k])
                    numCorrectTags += 1;
            }
        }
        float acc = float(numCorrectTags) / numTags * 100;
        if (acc > bestAcc) {
            std::printf("new best accuracy on eval set: %.2f%% (%d/%d)\n\n",
                        acc, numCorrectTags, numTags);
            bestAcc = acc;
        } else {
            std::printf("accuracy on eval set: %.2f%% (%d/%d)\n\n",
                        acc, numCorrectTags, numTags);
        }
    }
}
