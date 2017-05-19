//
// Created by Boliang Zhang on 5/17/17.
//

#ifndef PARSER_LOSS_H
#define PARSER_LOSS_H
#include <vector>
#include <random>
#include <Eigen>
#include "utils.h"

struct CrossEntropyCache {
    Eigen::MatrixXd pred;
    Eigen::MatrixXd ref;
    Eigen::MatrixXd loss;
};

struct CrossEntropyDiff {
    Eigen::MatrixXd pred_diff;
};


void crossEntropyForward(Eigen::MatrixXd & pred,
                         Eigen::MatrixXd & ref,
                         CrossEntropyCache & crossEntropyCache
                         ){
    Eigen::MatrixXd loss = - ref.array() * pred.array().log();

    crossEntropyCache.pred = pred;
    crossEntropyCache.ref = ref;
    crossEntropyCache.loss = loss;
}


void crossEntropyBackward(CrossEntropyCache & crossEntropyCache,
                          CrossEntropyDiff & crossEntropyDiff
                          ){
    Eigen::MatrixXd pred = crossEntropyCache.pred;
    Eigen::MatrixXd ref = crossEntropyCache.ref;

    Eigen::MatrixXd dpred = - ref.array() / pred.array();

    crossEntropyDiff.pred_diff = dpred;
}

void inputGradientCheck(CrossEntropyCache & crossEntropyCache,
                        Eigen::MatrixXd & inputGrad){
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

void crossEntropyGradientCheck(CrossEntropyCache & crossEntropyCache,
                               CrossEntropyDiff & crossEntropyDiff
                               ){
    std::cout << "=> gradient checking pred" << std::endl;
    inputGradientCheck(crossEntropyCache, crossEntropyDiff.pred_diff);
}

#endif //PARSER_LOSS_H
