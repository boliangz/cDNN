//
// Created by Boliang Zhang on 6/10/17.
//

#ifndef CDNN_SEQ2SEQ_H
#define CDNN_SEQ2SEQ_H

#include "../net.h"

struct InputSeq2Seq: Input {
    std::vector<int> srcIndex;
    Eigen::MatrixXd srcEmb;
    std::vector<int> trgIndex;
    Eigen::MatrixXd trgEmb;
    Eigen::MatrixXd trgOneHot;
    int seqLen;
};

class SeqToSeq: public Net {
public:
    SeqToSeq(const std::map<std::string, std::string> & configuration);
    SeqToSeq(const std::map<std::string, std::string> & configuration,
             const std::map<std::string, Eigen::MatrixXd*>& parameters);

//    void forward(const InputSeq2Seq & input);
    void forward(const InputSeq2Seq & input, bool isTrain);

    void backward();

    void gradientCheck(InputSeq2Seq & input);

    void update();

    std::vector<LSTM*> encoder;
    std::vector<LSTM*> decoder;
    MLP* mlp;
    CrossEntropyLoss* crossEntropyLoss;
};
#endif //CDNN_SEQ2SEQ_H
