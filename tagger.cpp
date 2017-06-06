//
// Created by Boliang Zhang on 6/5/17.
//

#include "charBiLSTMNet.h"
#include <numeric>


int main(int argc, char* argv []) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " model_dir eval_bio output_bio" << std::endl;
        return 1;
    }

    std::string modelDir = argv[1];
    std::string evalFile = argv[2];
    std::string outputFile = argv[3];

    RAWDATA evalRawData;

    loadRawData(evalFile, evalRawData);

    // load model
    std::map<std::string, int> word2id, char2id, label2id;
    std::map<int, std::string> id2word, id2char, id2label;
    Eigen::MatrixXd wordEmbedding;
    Eigen::MatrixXd charEmbedding;
    std::map<std::string, std::string> configuration;
    std::map<std::string, Eigen::MatrixXd*> parameters;

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
    std::ifstream evalFileStream(evalFile);
    std::string line;
    std::ofstream outputFileStream(outputFile);

    int numSeqToReport = 1000;
    for (int i = 0; i < evalData.size(); ++i ) {
        Sequence input = evalData[i];
        processData(input, wordEmbedding, charEmbedding);

        bool isTrain = false;
        charBiLSTMNet.forward(input, isTrain);
        Eigen::MatrixXd pred = charBiLSTMNet.mlp->output;

        std::vector<int> predLabelIndex;
        Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
        for (int k = 0; k < input.labelIndex.size(); ++k) {
            for (int l = 0; l < pred.rows(); ++l) {
                if (pred(l, k) == maxProba(k, 0))
                    predLabelIndex.push_back(l);
            }
        }

        for (int j = 0; j < predLabelIndex.size(); ++j) {
            std::getline(evalFileStream, line);
            outputFileStream << line
                             << " "
                             << id2label[predLabelIndex[j]]
                             << std::endl;
        }
        std::getline(evalFileStream, line);
        outputFileStream << std::endl;

        if ((i + 1) % numSeqToReport == 0){
            std::cout << i + 1 << " sequences processed." << std::endl;
        }
    }
    outputFileStream.close();
}
