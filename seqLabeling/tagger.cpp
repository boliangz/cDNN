//
// Created by Boliang Zhang on 6/5/17.
//

#include "charBiLSTMNet.h"
#include <numeric>


int main(int argc, char* argv []) {
    std::string evalFile = argv[1];
    std::string outputFile = argv[2];
    std::string modelDir = argv[3];

    RAWDATA evalRawData;

    loadRawData(evalFile, evalRawData, false);

    // load model
    std::map<std::string, int> word2id, char2id, label2id;
    std::map<int, std::string> id2word, id2char, id2label;
    Eigen::MatrixXd wordEmbedding;
    Eigen::MatrixXd charEmbedding;
    std::map<std::string, std::string> configuration;
    std::map<std::string, Eigen::MatrixXd*> parameters;

    std::cout << "=> loading Net..." << std::endl;
    CharBiLSTMNet::loadNet(modelDir, configuration, parameters,
                           word2id, char2id, label2id,
                           id2word, id2char, id2label,
                           wordEmbedding, charEmbedding);

    CharBiLSTMNet charBiLSTMNet(configuration, parameters);

    std::vector<Sequence> evalData;
    createData(evalRawData, word2id, char2id, label2id, evalData);


    //
    // starts tagging
    //
    std::cout << "=> start tagging..." << std::endl;
    std::ofstream outputFileStream(outputFile);

    int numSeqToReport = 500;
    for (int i = 0; i < evalData.size(); ++i ) {
        Sequence input = evalData[i];
        processData(input, wordEmbedding, charEmbedding);

        charBiLSTMNet.forward(input, false);
        Eigen::MatrixXd pred = charBiLSTMNet.mlp->output;

        std::vector<int> predLabelIndex;
        Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
        for (int k = 0; k < input.labelIndex.size(); ++k) {
            for (int l = 0; l < pred.rows(); ++l) {
                if (pred(l, k) == maxProba(k, 0))
                    predLabelIndex.push_back(l);
            }
        }

        std::vector<std::string> tokenInfo = evalRawData[i]["tokenInfo"];
        for (int j = 0; j < predLabelIndex.size(); ++j) {
            outputFileStream << tokenInfo[j]
                             << " "
                             << id2label[predLabelIndex[j]]
                             << std::endl;
        }
        outputFileStream << std::endl;

        if ((i + 1) % numSeqToReport == 0){
            std::cout << i + 1 << " sequences processed." << std::endl;
        }
    }
    outputFileStream.close();
}
