//
// Created by Boliang Zhang on 5/19/17.
//

#include <string>
#include <Eigen/Core>
#include <fstream>
#include <set>
#include <regex>
#include <iostream>
#include <map>
#include "loader.h"


void loadPreEmbedding(std::string & filePath,
                      std::map<std::string, Eigen::MatrixXd> & preEmbedding) {
    std::ifstream indata;
    indata.open(filePath);
    std::string line;
    std::vector<std::string> lineValues;
    uint rows = 0;
    while (std::getline(indata, line)) {
        if (rows == 0) {
            rows ++;
            continue;
        }
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        std::string word = lineValues[0];
        std::vector<double> emb;
        for (int i = 1; i < lineValues.size(); ++i) {
            emb.push_back(std::stod(lineValues[i]));
        }
        preEmbedding[word] = Eigen::MatrixXd::Map(emb.data(), emb.size(), 1);

        lineValues.clear();

        ++rows;
    }
}


void loadRawData(std::string & filePath,
                 RAWDATA & rawData){
    std::ifstream indata(filePath);
    std::string line;
    std::vector<std::string> lineValues;
    std::map<std::string, std::vector<std::string> > sequence;
    sequence["word"] = std::vector<std::string>();
    sequence["label"] = std::vector<std::string>();
    while (std::getline(indata, line)) {
        if (line.empty() || line[0] == '#') {
            if (not sequence["word"].empty() ||
                not sequence["label"].empty()) {
                rawData.push_back(sequence);
                sequence["word"].clear();
                sequence["label"].clear();
            }
        }
        else {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ' ')) {
                lineValues.push_back(cell);
            }
            std::string word = lineValues[0];
            std::string label = lineValues.back();
            sequence.find("word")->second.push_back(word);
            sequence.find("label")->second.push_back(label);
            lineValues.clear();
        }
    }
}


void set2map(const std::set<std::string> & s,
             std::map<int, std::string> & id2t,
             std::map<std::string, int> & t2id,
             bool addUNK) {
    int index;
    if (addUNK) {
        id2t[0] = "<UNK>";
        t2id["<UNK>"] = 0;
        index = 1;
    } else {
        index = 0;
    }

    std::vector<std::string> v(s.begin(), s.end());

    std::sort(v.begin(), v.end());

    std::vector<std::string>::iterator it;

    for ( it = v.begin(); it != v.end(); ++it) {
        id2t[index] = * it;
        t2id[* it] = index;
        index++;
    }
}


void createTokenSet(const RAWDATA & rawData,
                    std::set<std::string> & words,
                    std::set<std::string> & labels,
                    std::set<std::string> & chars) {
    for (int i = 0; i < rawData.size(); ++i) {
        int length = rawData[i].find("word")->second.size();
        for (int j = 0; j < length; ++j) {
            std::string w = rawData[i].find("word")->second[j];
            words.insert(w);

            for (char c: w) chars.insert(std::string{c});

            std::string l = rawData[i].find("label")->second[j];
            labels.insert(l);
        }
    }
}


void preEmbLookUp(Eigen::MatrixXd & wordEmbedding,
                  const std::map<std::string, Eigen::MatrixXd> & preEmbedding,
                  const std::map<int, std::string> & id2word) {
    int wordFound = 0;
    int wordLower = 0;
    int wordZeros = 0;
    for ( int i = 0; i < wordEmbedding.cols(); ++i ) {
        std::string w = id2word.find(i)->second;

        // create lowercase word
        std::string lowerWord = w;
        std::transform(w.begin(), w.end(), lowerWord.begin(), ::tolower);

        // replace digit by 0
        std::string lowerWordZero = lowerWord;
        for (int j = 0; j < lowerWordZero.size(); ++j) {
            if (std::isdigit(lowerWordZero[j])) lowerWordZero[j] = '0';
        }

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
    printf("%d / %d (%.2f%%) words have been initialized with pretrained embeddings.\n",
           wordFound+wordLower+wordZeros, int(id2word.size()),
           (wordFound+wordLower+wordZeros)/float(id2word.size())*100);
    printf("%i found directly, %i after lowercasing, %i after lowercasing + zero\n",
           wordFound, wordLower, wordZeros);
}


void createData(const RAWDATA & rawData,
                const std::map<std::string, int> & word2id,
                const std::map<std::string, int> & char2id,
                const std::map<std::string, int> & label2id,
                std::vector<Sequence> & data) {
    for ( int i = 0; i < rawData.size(); ++i ) {
        std::vector<std::string> words = rawData[i].find("word")->second;
        std::vector<std::string> labels = rawData[i].find("label")->second;

        std::vector<std::string>::iterator it;
        std::vector<int> wordIndex;
        std::vector<std::vector<int> > charIndex;
        for ( it = words.begin(); it != words.end(); ++it){
            if (word2id.find(*it) != word2id.end())
                wordIndex.push_back(word2id.find(*it)->second);
            else
                wordIndex.push_back(word2id.find("<UNK>")->second);

            std::vector<int> cIndex;
            for ( std::string::iterator sit = (*it).begin(); sit != (*it).end(); ++sit) {
                if (char2id.find(std::string(1, *sit)) != char2id.end())
                    cIndex.push_back(char2id.find(std::string(1, *sit))->second);
                else
                    cIndex.push_back(char2id.find("<UNK>")->second);
            }

            charIndex.push_back(cIndex);
        }

        std::vector<int> labelIndex;
        for ( it = labels.begin(); it != labels.end(); ++it)
            labelIndex.push_back(label2id.find(* it)->second);

        Sequence s;
        s.wordIndex = wordIndex;
        s.charIndex = charIndex;
        s.seqLen = wordIndex.size();

        // process label index
        s.labelIndex = labelIndex;
        s.labelOneHot = Eigen::MatrixXd::Constant(label2id.size(), s.seqLen, 0);
        for ( int j = 0; j < s.seqLen; ++j ) {
            s.labelOneHot(labelIndex[j], j) = 1;
        }

        data.push_back(s);
    }
}

void processData(Sequence & s,
                 const Eigen::MatrixXd& wordEmbedding,
                 const Eigen::MatrixXd& charEmbedding) {
    long wordEmbDim = wordEmbedding.rows();
    long charEmbDim = charEmbedding.rows();

    s.wordEmb = Eigen::MatrixXd::Constant(wordEmbDim, s.seqLen, 0);
    for (int i = 0; i < s.seqLen; i++) {
        s.wordEmb.col(i) = wordEmbedding.col(s.wordIndex[i]);

        long charLen = s.charIndex[i].size();
        Eigen::MatrixXd m(charEmbDim, charLen);
        for (int j = 0; j < charLen; ++j) {
            m.col(j) = charEmbedding.col(s.charIndex[i][j]);
        }
        s.charEmb.push_back(m);
    }
}

void expandWordSet(std::set<std::string> & trainWords,
                   const std::set<std::string> & evalWords,
                   const std::map<std::string, Eigen::MatrixXd> & preEmbedding) {
    for (std::set<std::string>::iterator it = evalWords.begin(); it != evalWords.end(); ++it) {
        std::string w = *it;

        // create lowercase word
        std::string lowerWord = w;
        std::transform(w.begin(), w.end(), lowerWord.begin(), ::tolower);

        // replace digit by 0
        std::string lowerWordZero = lowerWord;
        for (int i = 0; i < lowerWordZero.size(); ++i) {
            if (std::isdigit(lowerWordZero[i])) lowerWordZero[i] = '0';
        }

        if (trainWords.find(w) == trainWords.end() &&
            preEmbedding.find(w) != preEmbedding.end())
            trainWords.insert(*it);
        else if (trainWords.find(lowerWord) == trainWords.end() &&
                 preEmbedding.find(lowerWord) != preEmbedding.end())
            trainWords.insert(*it);
        else if (trainWords.find(lowerWordZero) == trainWords.end() &&
                 preEmbedding.find(lowerWordZero) != preEmbedding.end())
            trainWords.insert(*it);
    }
}

