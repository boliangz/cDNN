//
// Created by Boliang Zhang on 6/13/17.
//
#include "loader.h"
#include <fstream>


void loadRawData(const std::string & filePath,
                 const std::map<std::string, int>& srcToken2Id,
                 const std::map<std::string, int>& trgToken2Id,
                 std::vector<Seq2SeqInput> & data,
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

        Seq2SeqInput s;

        // process src sequence
        std::stringstream srcStream(lineValues[0]);
        std::vector<int> srcIndex;
        while (std::getline(srcStream, cell, ' '))
            srcIndex.push_back(srcToken2Id.at(cell));
        s.srcIndex = srcIndex;
        s.srcLen = s.srcIndex.size();

        // process trg sequence
        if (isTrain){
            std::stringstream trgStream(lineValues[1]);
            std::vector<int> trgIndex;
            while (std::getline(trgStream, cell, ' '))
                trgIndex.push_back(trgToken2Id.at(cell));
            s.trgIndex = trgIndex;
            s.trgLen = trgIndex.size();

            // process decoder label one-hot vectors
            s.trgOneHot = Eigen::MatrixXd::Constant(trgToken2Id.size(),
                                                    s.trgLen-1,
                                                    0);
            for ( int j = 0; j < s.trgLen-1; ++j ) {
                s.trgOneHot(s.trgIndex[j+1], j) = 1;
            }
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

    std::ifstream ifsSrcToken2Id(modelDir + "/model/src_token2id.mdl");
    while (std::getline(ifsSrcToken2Id, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        srcToken2Id[lineValues[0]] = std::stoi(lineValues[1]);
    }

    std::ifstream ifsSrcId2Token(modelDir + "/model/src_id2token.mdl");
    while (std::getline(ifsSrcId2Token, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        srcId2Token[std::stoi(lineValues[0])] = lineValues[1];
    }

    std::ifstream ifsTrgToken2Id(modelDir + "/model/trg_token2id.mdl");
    while (std::getline(ifsTrgToken2Id, line)) {
        std::vector<std::string> lineValues;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        trgToken2Id[lineValues[0]] = std::stoi(lineValues[1]);
    }

    std::ifstream ifsTrgId2Token(modelDir + "/model/trg_id2token.mdl");
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

//void createDate(std::vector<Seq2SeqInput> & data,
//                const std::map<std::string, int>& trgToken2Id){
//    for (int i = 0; i < data.size(); ++i) {
//        Seq2SeqInput& s = data[i];
//        // process decoder label one-hot vectors
//        s.trgOneHot = Eigen::MatrixXd::Constant(trgToken2Id.size(),
//                                                s.seqLen-1, 0);
//        for ( int j = 0; j < s.seqLen-1; ++j ) {
//            s.trgOneHot(s.trgIndex[j+1], j) = 1;
//        }
//    }
//}

void processData(Seq2SeqInput & s,
                 const Eigen::MatrixXd& srcEmbedding,
                 const Eigen::MatrixXd& trgEmbedding,
                 bool isTrain){
    // set target token embedding lookup dict
    s.trgTokenEmb = &trgEmbedding;

    long embDim = srcEmbedding.rows();

    s.srcEmb = Eigen::MatrixXd::Constant(embDim, s.srcLen, 0);
    // remove </s> in decoder input
    s.trgEmb = Eigen::MatrixXd::Constant(embDim, s.trgLen-1, 0);

    for (int i = 0; i < s.srcLen; i++) {
        s.srcEmb.col(i) = srcEmbedding.col(s.srcIndex[i]);
    }
    if (isTrain)
        for (int i = 0; i < s.trgLen - 1; i++) {
            s.trgEmb.col(i) = trgEmbedding.col(s.trgIndex[i]);
        }
    else
        // for test, if not target sentence provided, set first trgEmb to the emb
        // of <s>
        s.trgEmb = trgEmbedding.col(0);
}