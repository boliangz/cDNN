#include <iostream>
#include "Eigen"
#include "utils.h"
#include "bi_lstm_with_char.h"
#include "loader.h"


int main() {
//    if (argc < 3) {
//        std::cerr << "Usage: " << argv[0] << " TRAIN_BIO EVAL_BIO MODEL_DIR" << std::endl;
//        return 1;
//    }

    int wordDim = 50;
    int charDim = 25;
    int wordLSTMHiddenDim = 100;
    int charLSTMHiddenDim = 25;
    double learningRate = 0.01;
    double dropoutRate = 0.5;

//    std::string trainFile = argv[1];
//    std::string evalFile = argv[2];
//    std::string modelDir = argv[3];
    std::string trainFile = "/Users/boliangzhang/Documents/Phd/cDNN/data/updated_UD_English/en-ud-dev.bio";
    std::string evalFile = "/Users/boliangzhang/Documents/Phd/cDNN/data/updated_UD_English/en-ud-dev.bio";
    std::string modelDir = "/Users/boliangzhang/Documents/Phd/cDNN/model/";

    RAWDATA trainRawData;
    RAWDATA evalRawData;
    loadRawData(trainFile, trainRawData);
    loadRawData(evalFile, evalRawData);

    std::set<std::string> trainWords, trainLabels, trainChars, evalWords, evalLabels, evalChars;
    createTokenSet(trainRawData, trainWords, trainLabels, trainChars);
    createTokenSet(evalRawData, evalWords, evalLabels, evalChars);

    std::map<int, std::string> id2word, id2char, id2label;
    std::map<std::string, int> word2id, char2id, label2id;
    trainWords.insert(evalWords.begin(), evalWords.end());
    set2map(trainWords, id2word, word2id);
    trainChars.insert(evalChars.begin(), evalChars.end());
    set2map(trainChars, id2char, char2id);
    trainLabels.insert(evalLabels.begin(), evalLabels.end());
    set2map(trainLabels, id2label, label2id);

    std::vector<Sequence> trainData;
    std::vector<Sequence> evalData;
    createData(trainRawData, word2id, char2id, label2id, trainData);
    createData(evalRawData, word2id, char2id, label2id, evalData);

    std::string preEmbeddingFile = "/Users/boliangzhang/Documents/Phd/LORELEI/data/name_taggers/dnn/embeddings/eng_senna.emb";
    std::map<std::string, Eigen::MatrixXd> preEmbedding;
    std::printf("loading pre-trained embedding from: %s \n", preEmbeddingFile.c_str());
    loadPreEmbedding(preEmbeddingFile, preEmbedding);

    Eigen::MatrixXd wordEmbedding = initializeVariable(wordDim, word2id.size());
    preEmbLookUp(wordEmbedding, preEmbedding, id2word);

    Eigen::MatrixXd charEmbedding = initializeVariable(charDim, char2id.size());

    biLSTMCharRun(trainData, evalData, wordEmbedding, charEmbedding);

    return 0;
}