//
// Created by Boliang Zhang on 5/17/17.
//

#ifndef PARSER_DROPOUT_H
#define PARSER_DROPOUT_H

#include <vector>
#include <random>
#include <Eigen>
#include "utils.h"

struct DropoutCache {
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
    Eigen::MatrixXd mask;
};

struct DropoutDiff {
    Eigen::MatrixXd x_diff;
};


void dropoutForward(Eigen::MatrixXd & x,
                    double dropoutRate,
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


void dropoutBackward(Eigen::MatrixXd & dy,
                     DropoutCache & dropoutCache,
                     DropoutDiff & dropoutDiff
                     ){
    dropoutDiff.x_diff = dy.array() * dropoutCache.mask.array();
}


void inputGradCheck(Eigen::MatrixXd & dy,
                    DropoutCache & dropoutCache,
                    Eigen::MatrixXd & inputGrad
                    ){
    Eigen::MatrixXd & x = dropoutCache.x;

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


void dropoutGradientCheck(Eigen::MatrixXd & dy,
                          DropoutCache & dropoutCache,
                          DropoutDiff & dropoutDiff
){
    std::cout << "=> gradient checking x" << std::endl;
    inputGradCheck(dy, dropoutCache, dropoutDiff.x_diff);
}

#endif //PARSER_DROPOUT_H
