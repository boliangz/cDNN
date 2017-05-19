//
// Created by Boliang Zhang on 5/19/17.
//

#include <string>
#include <Eigen>
#include <fstream>
#include <set>
#include <regex>
#include "loader.h"


typedef std::vector<std::map<std::string, std::vector<std::string>>> DATA;

template<typename M>
void loadEmbedding(std::string filePath, M & embedding) {
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    embedding = Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}


void loadPreEmbedding(std::string filePath,
                      std::map<std::string, Eigen::MatrixXd> & preEmbedding
) {
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    std::vector<std::string> lineValues;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            lineValues.push_back(cell);
        }
        std::string word = lineValues[0];
        std::vector<double> emb;
        for (int i = 1; i < lineValues.size(); ++i) {
            emb.push_back(std::stod(lineValues[i]));
        }
        preEmbedding[word] = Eigen::VectorXd(emb.data());

        lineValues.clear();

        ++rows;
    }
}

void loadRawData(std::string filePath,
                 DATA & rawData
){
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    std::vector<std::string> lineValues;
    std::map<std::string, std::vector<std::string>> sequence;
    sequence["word"] = std::vector<std::string>();
    sequence["label"] = std::vector<std::string>();
    while (std::getline(indata, line)) {
        if (line.empty()) {
            rawData.push_back(sequence);
            sequence["word"].clear();
            sequence["label"].clear();
        }
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        sequence.find("word")->second.push_back(*lineValues.begin());
        sequence.find("label")->second.push_back(*lineValues.end());
    }
}

template <typename T>
void set2Map(const std::set<T> & s,
             std::map<int, std::string> & id2T,
             std::map<std::string, int> & t2Id
) {
    id2T[0] = "<UNK>";
    t2Id["<UNK>"] = 0;
    auto it;
    int index = 1;

    for ( it = s.begin(); it != s.end(); ++it) {
        id2T[index] = * it;
        t2Id[* it] = index;
    }
}

void generateTokenSet(const DATA & rawData,
                      std::set<std::string> & words,
                      std::set<std::string> & labels,
                      std::set<char> & chars
) {
    for (int i = 0; i < rawData.size(); ++i) {
        for (int j = 0; j < rawData[i].size(); ++j) {
            std::string w = rawData[i].find("word")->second[j];
            words.insert(w);

            std::set<char> c_set(w.c_str(), w.c_str() + w.size() + 1);
            chars.insert(c_set.begin(), c_set.end());

            std::string l = rawData[i].find("label")->second[j];
            labels.insert(l);
        }
    }
}

void embeddingLookUp(Eigen::MatrixXd & wordEmbedding,
                     std::map<std::string, Eigen::MatrixXd> & preEmbedding,
                     std::map<int, std::string> id2word
) {
    int c_found = 0;
    int c_lower = 0;
    int c_zeros = 0;
    for ( int i = 0; i < wordEmbedding.cols(); ++i ) {
        std::string w = id2word.find(i)->second;

        // create lowercase word
        std::string lowerWord = w;
        std::transform(lowerWord.begin(), lowerWord.end(), lowerWord, std::tolower);

        // replace digit by 0
        std::regex r("\\d");
        std::string lowerWordZero;
        std::regex_replace(lowerWordZero, lowerWord.begin(), lowerWord.end(), r, "0");


        if ( preEmbedding.find(w) != preEmbedding.end()) {  // word found in pretrained embeddings
            wordEmbedding.col(i) = preEmbedding.find(w)->second;
        }
        else {
            std::string lowerWord = w;
            std::transform(lowerWord.begin(), lowerWord.end(), lowerWord, std::tolower);
            if ( preEmbedding.find(lowerWord) != preEmbedding.end() ) {
                wordEmbedding.col(i) = preEmbedding.find(lowerWord)->second;
            }

        }

    }

}

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




void preprocessData(DATA & rawData,
                    DATA & ) {

}

