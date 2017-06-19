//
// Created by Boliang Zhang on 6/10/17.
//

#ifndef CDNN_SEQ2SEQ_H
#define CDNN_SEQ2SEQ_H

#include "../net.h"

struct Seq2SeqInput: public Input {
    std::vector<int> srcIndex;
    Eigen::MatrixXd srcEmb;
    std::vector<int> trgIndex;
    Eigen::MatrixXd trgEmb;
    Eigen::MatrixXd trgOneHot;
    // target token embedding lookup dict, used in predicting
    const Eigen::MatrixXd* trgTokenEmb;
    int srcLen;
    int trgLen;
};

class SeqToSeq: public Net {
public:
    SeqToSeq(const std::map<std::string, std::string> & configuration);
    SeqToSeq(const std::map<std::string, std::string> & configuration,
             const std::map<std::string, Eigen::MatrixXd*>& parameters);

//    void forward(const Seq2SeqInput & input);
    void forward(const Input & input, bool isTrain);

    void backward();

    void gradientCheck(Seq2SeqInput & input);

    void update();

    std::vector<LSTM*> encoder;
    std::vector<LSTM*> decoder;
    MLP* mlp;
    CrossEntropyLoss* crossEntropyLoss;
};
#endif //CDNN_SEQ2SEQ_H
