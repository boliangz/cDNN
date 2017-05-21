//
// Created by Boliang Zhang on 5/18/17.
//

#ifndef PARSER_LOADER_H
#define PARSER_LOADER_H

#include <string>
#include <Eigen>
#include <fstream>
#include <set>
#include <regex>


struct Sequence {
    std::vector<int> wordIndex;
    Eigen::MatrixXd wordEmb;
    std::vector<std::vector<int>> charIndex;
    std::vector<Eigen::MatrixXd> charEmb;
    std::vector<int> labelIndex;
    Eigen::MatrixXd labelOneHot;
    int seqLen;
};

typedef std::vector<std::map<std::string, std::vector<std::string>>> DATA;

template<typename M>
void loadEmbedding(std::string filePath, M & embedding);


void loadPreEmbedding(std::string filePath, std::map<std::string, Eigen::MatrixXd> & preEmbedding);

void loadRawData(std::string filePath, DATA & rawData);

template <typename T>
void set2map(const std::set<T> & s, std::map<int, std::string> & id2t, std::map<std::string, int> & t2id);

void generateTokenSet(const DATA & rawData, std::set<std::string> & words, std::set<std::string> & labels,
                      std::set<char> & chars);

void parsePreEmbedding(Eigen::MatrixXd & wordEmbedding, std::map<std::string, Eigen::MatrixXd> & preEmbedding,
                     std::map<int, std::string> id2word);

//template <class T>
//void generateMapping(std::set<std::string> & s,
//                     std::map<int, T> id2T,
//                     std::map<T, int> T2id,
//                     ) {
//
//    std::map<int, std::string> id2word;
//    std::map<int, std::string> id2char;
//    std::map<int, std::string> id2label;
//    std::map<std::string, int> word2id;
//    std::map<std::string, int> char2id;
//    std::map<std::string, int> label2id;
//
//    set2Map<std::string>(words, id2word, word2id);
//    set2Map<std::string>(labels, id2label, label2id);
//    set2Map<char>(chars, id2char, char2id);
//
//}


void preprocessData(DATA & rawData, DATA & );


#endif //PARSER_LOADER_H
