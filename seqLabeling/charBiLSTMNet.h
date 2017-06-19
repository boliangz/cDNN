//
// Created by Boliang Zhang on 5/31/17.
//

#ifndef CDNN_CHARBILSTMNET_H
#define CDNN_CHARBILSTMNET_H

#include "../net.h"

struct SeqLabelingInput: public Input {
    std::vector<int> wordIndex;
    Eigen::MatrixXd wordEmb;
    std::vector<std::vector<int> > charIndex;
    std::vector<Eigen::MatrixXd> charEmb;
    std::vector<int> labelIndex;
    Eigen::MatrixXd labelOneHot;
    int seqLen;
};

class CharBiLSTMNet: public Net {
public:
    CharBiLSTMNet(){}
    CharBiLSTMNet(const std::map<std::string, std::string> & configuration);
    CharBiLSTMNet(const std::map<std::string, std::string> & configuration,
                  const std::map<std::string, Eigen::MatrixXd*>& parameters);

//    void forward(const SeqLabelingInput & input);
    void forward(const Input & input, bool isTrain);

    void backward();

    void update();

    void gradientCheck(SeqLabelingInput& input);

    LSTM* charFwdLSTM;
    LSTM* charBwdLSTM;
    BiLSTM* wordBiLSTM;
    MLP* mlp;
    Dropout* dropout;
    CrossEntropyLoss* crossEntropyLoss;
};

#endif //CDNN_CHARBILSTMNET_H
