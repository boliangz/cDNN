//
// Created by Boliang Zhang on 6/13/17.
//
#include "loader.h"
#include <fstream>


void loadRawData(std::string & filePath,
                 std::vector<InputSeq2Seq> & data,
                 bool isTrain){
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    while (std::getline(indata, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, '\t')) {
            lineValues.push_back(cell);
        }

        InputSeq2Seq s;

        // process src sequence
        std::stringstream srcStream(lineValues[0]);
        std::vector<int> srcIndex;
        while (std::getline(srcStream, cell, ' '))
            srcIndex.push_back(std::stoi(cell));
        s.srcIndex = srcIndex;

        // process trg sequence
        if (isTrain){
            std::stringstream trgStream(lineValues[1]);
            std::vector<int> trgIndex;
            while (std::getline(trgStream, cell, ' '))
                trgIndex.push_back(std::stoi(cell));
            s.trgIndex = trgIndex;
        }

        data.push_back(s);
    }
}

void loadMapping(std::string & modelDir,
                 std::map<std::string, int> & srcToken2Id,
                 std::map<int, std::string> & srcId2Token,
                 std::map<std::string, int> & trgToken2Id,
                 std::map<int, std::string> & trgId2Token){
    std::string line;

    std::ifstream ifsSrcToken2Id(modelDir + "/src_token2id.mdl");
    while (std::getline(ifsSrcToken2Id, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        srcToken2Id[lineValues[0]] = std::stoi(lineValues[1]);
    }

    std::ifstream ifsSrcId2Token(modelDir + "/src_id2token.mdl");
    while (std::getline(ifsSrcId2Token, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        srcId2Token[std::stoi(lineValues[0])] = lineValues[1];
    }

    std::ifstream ifsTrgToken2Id(modelDir + "/trg_token2id.mdl");
    while (std::getline(ifsTrgToken2Id, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        trgToken2Id[lineValues[0]] = std::stoi(lineValues[1]);
    }

    std::ifstream ifsTrgId2Token(modelDir + "/trg_id2token.mdl");
    while (std::getline(ifsTrgId2Token, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        trgId2Token[std::stoi(lineValues[0])] = lineValues[1];
    }
}

void createDate(std::vector<InputSeq2Seq> & data,
                const std::map<std::string, int>& trgToken2Id){
    for (int i = 0; i < data.size(); ++i) {
        InputSeq2Seq& s = data[i];
        // process decoder label one-hot vectors
        s.trgOneHot = Eigen::MatrixXd::Constant(trgToken2Id.size(),
                                                s.seqLen-1, 0);
        for ( int j = 0; j < s.seqLen-1; ++j ) {
            s.trgOneHot(s.trgIndex[j+1], j) = 1;
        }
    }
}

void processData(InputSeq2Seq & s,
                 const Eigen::MatrixXd& srcEmbedding,
                 const Eigen::MatrixXd& trgEmbedding){
    long embDim = srcEmbedding.rows();

    s.srcEmb = Eigen::MatrixXd::Constant(embDim, s.seqLen, 0);
    // remove </s> in decoder input
    s.trgEmb = Eigen::MatrixXd::Constant(embDim, s.seqLen-1, 0);

    for (int i = 0; i < s.seqLen; i++) {
        s.srcEmb.col(i) = srcEmbedding.col(s.srcIndex[i]);
    }
    for (int i = 0; i < s.seqLen - 1; i++) {
        s.trgEmb.col(i) = trgEmbedding.col(s.trgIndex[i]);
    }
}