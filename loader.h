//
// Created by Boliang Zhang on 5/18/17.
//

#ifndef CDNN_LOADER_H
#define CDNN_LOADER_H

#include <string>
#include <Eigen/Core>
#include <fstream>
#include <set>
#include <regex>
#include <map>


typedef std::vector<std::map<std::string, std::vector<std::string> > > RAWDATA;

struct Sequence {
    std::vector<int> wordIndex;
    Eigen::MatrixXd wordEmb;
    std::vector<std::vector<int> > charIndex;
    std::vector<Eigen::MatrixXd> charEmb;
    std::vector<int> labelIndex;
    Eigen::MatrixXd labelOneHot;
    int seqLen;
};


void loadPreEmbedding(std::string & filePath, std::map<std::string, Eigen::MatrixXd> & preEmbedding);

void loadRawData(std::string & filePath, RAWDATA & rawData);

void set2map(const std::set<std::string> & s,
             std::map<int, std::string> & id2t,
             std::map<std::string, int> & t2id,
             bool addUNK);

void createTokenSet(const RAWDATA & rawData,
                    std::set<std::string> & words,
                    std::set<std::string> & labels,
                    std::set<std::string> & chars);

void preEmbLookUp(Eigen::MatrixXd & wordEmbedding,
                  const std::map<std::string, Eigen::MatrixXd> & preEmbedding,
                  const std::map<int, std::string> & id2word);

void createData(const RAWDATA & rawData,
                const std::map<std::string, int> & word2id,
                const std::map<std::string, int> & char2id,
                const std::map<std::string, int> & label2id,
                std::vector<Sequence> & data);

void processData(Sequence & s,
                 const Eigen::MatrixXd & wordEmbedding,
                 const Eigen::MatrixXd & charEmbedding);

void expandWordSet(std::set<std::string> & trainWords,
                   const std::set<std::string> & evalWords,
                   const std::map<std::string, Eigen::MatrixXd> & preEmbedding);

#endif //CDNN_LOADER_H
