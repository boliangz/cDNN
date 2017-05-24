#include <iostream>
#include "Eigen"
#include "utils.h"
#include "bi_lstm.h"
#include "loader.h"


int main(int argc, char* argv []) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " train_bio eval_bio pretrain_emb" << std::endl;
        return 1;
    }

    int wordDim = 50;
    int charDim = 25;

    std::string trainFile = argv[1];
    std::string evalFile = argv[2];
    std::string preEmbeddingFile = argv[3];
    std::string modelDir = argv[4];

    RAWDATA trainRawData;
    RAWDATA evalRawData;
    loadRawData(trainFile, trainRawData);
    loadRawData(evalFile, evalRawData);

    std::set<std::string> trainWords, trainLabels, trainChars, evalWords, evalLabels, evalChars;
    createTokenSet(trainRawData, trainWords, trainLabels, trainChars);
    createTokenSet(evalRawData, evalWords, evalLabels, evalChars);

    std::map<std::string, Eigen::MatrixXd> preEmbedding;
    std::printf("loading pre-trained embedding from: %s \n", preEmbeddingFile.c_str());
    loadPreEmbedding(preEmbeddingFile, preEmbedding);

    expandWordSet(trainWords, evalWords, preEmbedding);
    std::map<int, std::string> id2word, id2char, id2label;
    std::map<std::string, int> word2id, char2id, label2id;
//    trainWords.insert(evalWords.begin(), evalWords.end());
    set2map(trainWords, id2word, word2id);
    trainChars.insert(evalChars.begin(), evalChars.end());
    set2map(trainChars, id2char, char2id);
    trainLabels.insert(evalLabels.begin(), evalLabels.end());
    set2map(trainLabels, id2label, label2id);

    std::vector<Sequence> trainData;
    std::vector<Sequence> evalData;
    createData(trainRawData, word2id, char2id, label2id, trainData);
    createData(evalRawData, word2id, char2id, label2id, evalData);

    Eigen::MatrixXd wordEmbedding = initializeVariable(wordDim, word2id.size());
    preEmbLookUp(wordEmbedding, preEmbedding, id2word);

    Eigen::MatrixXd charEmbedding = initializeVariable(charDim, char2id.size());

//    trainData = std::vector<Sequence>(trainData.begin(), trainData.begin()+10);
//    evalData = std::vector<Sequence>(evalData.begin(), evalData.begin()+10);
    train(trainData, evalData, modelDir, wordEmbedding, charEmbedding);

    return 0;
}