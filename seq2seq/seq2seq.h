//
// Created by Boliang Zhang on 6/10/17.
//

#ifndef CDNN_SEQ2SEQ_H
#define CDNN_SEQ2SEQ_H

#include "../net.h"

class SeqToSeq: public Net {
public:
    SeqToSeq(const std::map<std::string, std::string> & configuration);
    SeqToSeq(const std::map<std::string, std::string> & configuration,
             const std::map<std::string, Eigen::MatrixXd*>& parameters);

    void forward(const Sequence & input);
    void forward(const Sequence & input, bool isTrain);

    void backward();

    void update();

    std::vector<BiLSTM*> encoder;
    std::vector<LSTM*> decoder;
};
#endif //CDNN_SEQ2SEQ_H
