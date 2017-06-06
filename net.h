//
// Created by Boliang Zhang on 5/30/17.
//

#ifndef CDNN_NET_H
#define CDNN_NET_H

#include "nn.h"
#include <Eigen/Core>


class Net: public Layer {
public:
    virtual void forward(const Sequence & input){}
    virtual void forward(const Sequence & input, bool isTrain){}

    static void saveNet(const std::map<std::string, std::string>& configuration,
                        const std::map<std::string, Eigen::MatrixXd*> parameters,
                        const std::map<std::string, int>& word2id,
                        const std::map<std::string, int>& char2id,
                        const std::map<std::string, int>& label2id,
                        const std::map<int, std::string>& id2word,
                        const std::map<int, std::string>& id2char,
                        const std::map<int, std::string>& id2label,
                        const Eigen::MatrixXd& wordEmbedding,
                        const Eigen::MatrixXd& charEmbedding);

    static void loadNet(std::string modelDir,
                        std::map<std::string, std::string>& configuration,
                        std::map<std::string, Eigen::MatrixXd*>& parameters,
                        std::map<std::string, int>& word2id,
                        std::map<std::string, int>& char2id,
                        std::map<std::string, int>& label2id,
                        std::map<int, std::string>& id2word,
                        std::map<int, std::string>& id2char,
                        std::map<int, std::string>& id2label,
                        Eigen::MatrixXd& wordEmbedding,
                        Eigen::MatrixXd& charEmbedding);

    void gradientCheck(Sequence & input);

    void updateEmbedding(Eigen::MatrixXd* embDict,
                         const std::vector<int> & wordIndex);

    std::map<std::string, std::string> configuration;
};

Eigen::MatrixXd readMatrix(const char *filename);

#endif //CDNN_NET_H
