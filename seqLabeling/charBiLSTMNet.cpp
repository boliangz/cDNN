//
// Created by Boliang Zhang on 5/31/17.
//
#include "charBiLSTMNet.h"
#include "../nn.h"

CharBiLSTMNet::CharBiLSTMNet(const std::map<std::string, std::string> & configuration) {
    this->configuration = configuration;

    int wordDim = std::stoi(this->configuration["wordDim"]);
    int charDim = std::stoi(this->configuration["charDim"]);
    int wordLSTMHiddenDim = std::stoi(this->configuration["wordLSTMHiddenDim"]);
    int charLSTMHiddenDim = std::stoi(this->configuration["charLSTMHiddenDim"]);
    float dropoutRate = std::stof(this->configuration["dropoutRate"]);
    int labelSize = std::stoi(this->configuration["labelSize"]);

    bool isBatch = true;
    charFwdLSTM = new LSTM(charDim, charLSTMHiddenDim, "charFwdLSTM", isBatch);
    charBwdLSTM = new LSTM(charDim, charLSTMHiddenDim, "charBwdLSTM", isBatch);
    wordBiLSTM = new BiLSTM(wordDim + 2 * charLSTMHiddenDim,
                            wordLSTMHiddenDim, "wordBiLSTM");
    mlp = new MLP(2 * wordLSTMHiddenDim, labelSize, "mlp");
    dropout = new Dropout(dropoutRate, "dropout");
    crossEntropyLoss = new CrossEntropyLoss("crossEntropyLoss");

    parameters.insert(charFwdLSTM->parameters.begin(),
                      charFwdLSTM->parameters.end());
    parameters.insert(charBwdLSTM->parameters.begin(),
                      charBwdLSTM->parameters.end());
    parameters.insert(wordBiLSTM->parameters.begin(),
                      wordBiLSTM->parameters.end());
    parameters.insert(mlp->parameters.begin(),
                      mlp->parameters.end());
}

CharBiLSTMNet::CharBiLSTMNet(const std::map<std::string, std::string> & configuration,
                             const std::map<std::string, Eigen::MatrixXd*>& parameters){
    this->configuration = configuration;

    int wordDim = std::stoi(this->configuration["wordDim"]);
    int charDim = std::stoi(this->configuration["charDim"]);
    int wordLSTMHiddenDim = std::stoi(this->configuration["wordLSTMHiddenDim"]);
    int charLSTMHiddenDim = std::stoi(this->configuration["charLSTMHiddenDim"]);
    float dropoutRate = std::stof(this->configuration["dropoutRate"]);
    int labelSize = std::stoi(this->configuration["labelSize"]);

    bool isBatch = true;
    charFwdLSTM = new LSTM(charDim, charLSTMHiddenDim, "charFwdLSTM", isBatch, parameters);
    charBwdLSTM = new LSTM(charDim, charLSTMHiddenDim, "charBwdLSTM", isBatch, parameters);

    wordBiLSTM = new BiLSTM(wordDim + 2 * charLSTMHiddenDim,
                            wordLSTMHiddenDim, "wordBiLSTM",
                            parameters);

    mlp = new MLP(2 * wordLSTMHiddenDim, labelSize, "mlp", parameters);
    dropout = new Dropout(dropoutRate, "dropout");
    crossEntropyLoss = new CrossEntropyLoss("crossEntropyLoss");

    this->parameters.insert(charFwdLSTM->parameters.begin(),
                            charFwdLSTM->parameters.end());
    this->parameters.insert(charBwdLSTM->parameters.begin(),
                            charBwdLSTM->parameters.end());
    this->parameters.insert(wordBiLSTM->parameters.begin(),
                            wordBiLSTM->parameters.end());
    this->parameters.insert(mlp->parameters.begin(),
                            mlp->parameters.end());
}


void CharBiLSTMNet::forward(const Sequence & input) {
    // configuration
    int wordDim = std::stoi(configuration["wordDim"]);
    int charDim = std::stoi(configuration["charDim"]);
    int wordLSTMHiddenDim = std::stoi(configuration["wordLSTMHiddenDim"]);
    int charLSTMHiddenDim = std::stoi(configuration["charLSTMHiddenDim"]);
    float dropoutRate = std::stoi(configuration["dropoutRate"]);
    int sequenceLen = input.seqLen;

    // bi-directional char lstm forward
    Eigen::MatrixXd sequcenCharEmb(2 * charLSTMHiddenDim, sequenceLen);

    charFwdLSTM->Layer::forward(input.charEmb);

    std::vector<Eigen::MatrixXd> reversedCharEmb;
    for (int i = 0; i < sequenceLen; ++i) {
        reversedCharEmb.push_back(input.charEmb[i].rowwise().reverse());
    }
    charBwdLSTM->Layer::forward(reversedCharEmb);

    for (int i = 0; i < sequenceLen; ++i ) {
        sequcenCharEmb.col(i) << charFwdLSTM->batchOutput[i].rightCols(1),
                charBwdLSTM->batchOutput[i].rightCols(1);
    }

    // dropout forward
    Eigen::MatrixXd dropoutInput(wordDim + 2 * charLSTMHiddenDim, sequenceLen);
    dropoutInput << input.wordEmb, sequcenCharEmb;  // concatenate word embedding and two character embeddings.
    dropout->forward(dropoutInput);

    // bi-directional word lstm forward
    wordBiLSTM->forward(dropout->output);

    // mlp forward
    mlp->forward(wordBiLSTM->output);

    // cross entropy forward
    crossEntropyLoss->forward(mlp->output, input.labelOneHot);

    cache[name+"_output"] = crossEntropyLoss->output;
    output = crossEntropyLoss->output;
}

void CharBiLSTMNet::forward(const Sequence & input, bool isTrain) {
    if (!isTrain) {
        double originalDropoutRate = dropout->dropoutRate;
        dropout->dropoutRate = 0;
        forward(input);
        dropout->dropoutRate = originalDropoutRate;
    } else {
        forward(input);
    }
}

void CharBiLSTMNet::backward() {
    // cross entropy backward
    crossEntropyLoss->backward();

    // mlp backward
    mlp->backward(crossEntropyLoss->inputDiff);

    // word lstm backward
    wordBiLSTM->backward(mlp->inputDiff);

    // dropout backward
    dropout->backward(wordBiLSTM->inputDiff);

    // char lstm backward
    int sequenceLen = mlp->output.cols();
    int wordDim = std::stoi(configuration["wordDim"]);
    int charLSTMHiddenDim = std::stoi(configuration["charLSTMHiddenDim"]);
    std::vector<Eigen::MatrixXd> batchCharFwdLSTMDy;
    std::vector<Eigen::MatrixXd> batchCharBwdLSTMDy;
    for (int i = 0; i < sequenceLen; i++) {
        int tokenLen = charFwdLSTM->batchOutput[i].cols();
        Eigen::MatrixXd charFwdLSTMDy(charLSTMHiddenDim, tokenLen);
        charFwdLSTMDy.setZero();
        charFwdLSTMDy.rightCols(1) =
                dropout->inputDiff.block(wordDim, i,
                                         charLSTMHiddenDim, 1);
        batchCharFwdLSTMDy.push_back(charFwdLSTMDy);

        Eigen::MatrixXd charBwdLSTMDy(charLSTMHiddenDim, tokenLen);
        charBwdLSTMDy.setZero();
        charBwdLSTMDy.rightCols(1) =
                dropout->inputDiff.block(wordDim+charLSTMHiddenDim, i,
                                         charLSTMHiddenDim, 1);
        batchCharBwdLSTMDy.push_back(charBwdLSTMDy);
    }
    charFwdLSTM->Layer::backward(batchCharFwdLSTMDy);
    charBwdLSTM->Layer::backward(batchCharBwdLSTMDy);


    diff.clear();
    diff.insert(charFwdLSTM->diff.begin(),
                charFwdLSTM->diff.end());
    diff.insert(charBwdLSTM->diff.begin(),
                charBwdLSTM->diff.end());
    diff.insert(wordBiLSTM->diff.begin(),
                wordBiLSTM->diff.end());
    diff.insert(mlp->diff.begin(),
                mlp->diff.end());
    diff[name+"_wordInput"] = dropout->inputDiff.topRows(wordDim);
}

void CharBiLSTMNet::update(){
    float learningRate = std::stof(configuration["learningRate"]);
    Layer::update(learningRate);
}