//
// Created by Boliang Zhang on 5/19/17.
//

#include <string>
#include <Eigen>
#include <fstream>
#include <set>
#include <regex>
#include <iostream>
#include "loader.h"


typedef std::vector<std::map<std::string, std::vector<std::string>>> RAWDATA;

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
                 RAWDATA & rawData
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
             std::map<int, std::string> & id2t,
             std::map<std::string, int> & t2id
) {
    id2t[0] = "<UNK>";
    t2id["<UNK>"] = 0;
    auto it;
    int index = 1;

    for ( it = s.begin(); it != s.end(); ++it) {
        id2t[index] = * it;
        t2id[* it] = index;
    }
}

void createTokenSet(const RAWDATA & rawData,
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

void preEmbLookUp(Eigen::MatrixXd & wordEmbedding,
                  const std::map<std::string, Eigen::MatrixXd> & preEmbedding,
                  const std::map<int, std::string> & id2word
                  ) {
    int wordFound = 0;
    int wordLower = 0;
    int wordZeros = 0;
    for ( int i = 0; i < wordEmbedding.cols(); ++i ) {
        std::string w = id2word.find(i)->second;

        // create lowercase word
        std::string lowerWord = w;
        std::transform(lowerWord.begin(), lowerWord.end(), lowerWord, std::tolower);

        // replace digit by 0
        std::regex r("\\d");
        std::string lowerWordZero = lowerWord;
        std::regex_replace(lowerWord, r, "0");

        if ( preEmbedding.find(w) != preEmbedding.end()) {  // word found in pretrained embeddings
            wordEmbedding.col(i) = preEmbedding.find(w)->second;
            wordFound += 1;
        }
        else if ( preEmbedding.find(lowerWord) != preEmbedding.end() ) {
            wordEmbedding.col(i) = preEmbedding.find(lowerWord)->second;
            wordLower += 1;
        }
        else if ( preEmbedding.find(lowerWordZero) != preEmbedding.end() ) {
            wordEmbedding.col(i) = preEmbedding.find(lowerWordZero)->second;
            wordZeros += 1;
        }
    }

    std::cout << preEmbedding.size() << " pre-trained word embeddings loaded." << std::endl;
    printf("%d / %d (%.4f%%) words have been initialized with pretrained embeddings.",
           wordFound+wordLower+wordZeros, id2word.size(),
           (wordFound+wordLower+wordZeros)/id2word.size()*100.);
    printf("%i found directly, %i after lowercasing, %i after lowercasing + zero",
            wordFound, wordLower, wordZeros);
}

void createData(const RAWDATA & rawData,
                const std::map<std::string, int> & word2id,
                const std::map<char, int> & char2id,
                const std::map<std::string, int> & label2id,
                std::vector<Sequence> data
                ) {
    for ( int i = 0; i < rawData.size(); ++i ) {
        std::vector<std::string>* words = &rawData[i].find("word")->second;
        std::vector<std::string>* labels = &rawData[i].find("label")->second;

        std::vector<std::string>::iterator it;
        std::vector<int> wordIndex;
        std::vector<std::vector<int>> charIndex;
        for ( it = words->begin(); it != words->end(); ++it){
            wordIndex.push_back(word2id.find(*it)->second);
            std::vector<int> cIndex;
            for ( std::string::iterator sit = (*it).begin(); sit != (*it).end(); ++sit)
                cIndex.push_back(char2id.find(*sit)->second);
            charIndex.push_back(cIndex);
        }

        std::vector<int> labelIndex;
        for ( it = labels->begin(); it != labels->end(); ++it)
            labelIndex.push_back(label2id.find(* it)->second);

        Sequence s;
        s.wordIndex = wordIndex;
        s.charIndex = charIndex;
        s.labelIndex = labelIndex;
        data.push_back(s);
    }
}

