//
// Created by Boliang Zhang on 5/31/17.
//

#ifndef CDNN_CHARBILSTMNET_H
#define CDNN_CHARBILSTMNET_H

#include "../net.h"

class CharBiLSTMNet: public Net {
public:
    CharBiLSTMNet(){}
    CharBiLSTMNet(const std::map<std::string, std::string> & configuration);
    CharBiLSTMNet(const std::map<std::string, std::string> & configuration,
                  const std::map<std::string, Eigen::MatrixXd*>& parameters);

    void forward(const Sequence & input);
    void forward(const Sequence & input, bool isTrain);

    void backward();

    void update();

    LSTM* charFwdLSTM;
    LSTM* charBwdLSTM;
    BiLSTM* wordBiLSTM;
    MLP* mlp;
    Dropout* dropout;
    CrossEntropyLoss* crossEntropyLoss;
};

#endif //CDNN_CHARBILSTMNET_H
