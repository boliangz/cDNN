#include "charBiLSTMNet.h"
#include "../utils.h"
#include "loader.h"
#include <iostream>
#include <numeric>
#include <Eigen/Core>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>

std::string conllScorer = "/nas/data/m1/zhangb8/DNN/cdnn/conlleval";
//std::string conllScorer = "/Users/boliangzhang/Documents/Phd/DNN/cDNN/conlleval";

struct ThreadArgs {
    std::vector<Sequence>* trainData;
    Eigen::MatrixXd* wordEmbedding;
    Eigen::MatrixXd* charEmbedding;
    std::vector<int>* index;
    std::map<std::string, std::string>* configuration;
    std::map<std::string, Eigen::MatrixXd*>* parameters;
};

void* mainThread(void* threadarg){
    struct ThreadArgs *args = (struct ThreadArgs *) threadarg;

    std::vector<Sequence>* trainData = args->trainData;
    Eigen::MatrixXd* wordEmbedding = args->wordEmbedding;
    Eigen::MatrixXd* charEmbedding = args->charEmbedding;
    std::vector<int>* index = args->index;
    std::map<std::string, Eigen::MatrixXd*>* parameters = args->parameters;

    // initialize net
    CharBiLSTMNet charBiLSTMNet(*args->configuration, *parameters);

    int numSeqToReport = 500;
    std::vector<float> epoch_loss;
    for (int j = 0; j < index->size(); ++j){
        Sequence input = (*trainData)[(*index)[j]-1];

        processData(input, *wordEmbedding, *charEmbedding);
        //
        // network forward
        //
        charBiLSTMNet.forward(input);
        Eigen::MatrixXd loss = charBiLSTMNet.crossEntropyLoss->output;
        epoch_loss.push_back(loss.sum() / input.seqLen);

        //
        // network backward
        //
        charBiLSTMNet.backward();

        //
        // network gradient check
        //
//            charBiLSTMNet.gradientCheck(input);

        //
        // network parameters update
        //
        charBiLSTMNet.update();
        charBiLSTMNet.updateEmbedding(wordEmbedding, input.wordIndex);

        if ((j + 1) % numSeqToReport == 0){
            std::vector<float> v(epoch_loss.end() - numSeqToReport, epoch_loss.end());
            float average = std::accumulate( v.begin(), v.end(), 0.0)/v.size();
            std::cout << j + 1 << " sequences loss: " << average << std::endl;
        }
    }
}


int main(int argc, char* argv []) {
    // parse argument
    std::string trainFile = argv[1];
    std::string evalFile = argv[2];
    std::string modelDir = argv[3];
    std::string preEmbeddingFile = argv[4];
    std::string wordDim = argv[5];
    std::string charDim = argv[6];
    std::string wordLSTMHiddenDim = argv[7];
    std::string charLSTMHiddenDim = argv[8];
    std::string learningRate = argv[9];
    std::string dropoutRate = argv[10];
    std::string allEmb = argv[11];
    std::string numThread = argv[12];
    std::string conllScorer = argv[13];

    std::map<std::string, std::string> netConf =
            {
                    {"wordDim", wordDim},
                    {"charDim", charDim},
                    {"wordLSTMHiddenDim", wordLSTMHiddenDim},
                    {"charLSTMHiddenDim", charLSTMHiddenDim},
                    {"learningRate", learningRate},
                    {"dropoutRate", dropoutRate},
                    {"preEmb", preEmbeddingFile},
                    {"numThreads", numThread},
                    {"modelDir", modelDir},
                    {"allEmb", allEmb},
                    {"conllScorer", conllScorer}
            };

    // load raw data
    RAWDATA trainRawData;
    RAWDATA evalRawData;
    loadRawData(trainFile, trainRawData, true);
    loadRawData(evalFile, evalRawData, true);

    // create word, label, char set
    std::set<std::string> trainWords, trainLabels, trainChars,
            evalWords, evalLabels, evalChars;
    createTokenSet(trainRawData, trainWords, trainLabels, trainChars);
    createTokenSet(evalRawData, evalWords, evalLabels, evalChars);

    // load pre-trained embeddings
    std::map<std::string, Eigen::MatrixXd> preEmbedding;
    if (preEmbeddingFile != "0") {
        std::printf("=> loading pre-trained embedding from: %s \n",
                    preEmbeddingFile.c_str());
        loadPreEmbedding(preEmbeddingFile, preEmbedding);
        expandWordSet(trainWords, evalWords, preEmbedding);
    }

    std::map<int, std::string> id2word, id2char, id2label;
    std::map<std::string, int> word2id, char2id, label2id;
    set2map(trainWords, id2word, word2id, true);
    trainChars.insert(evalChars.begin(), evalChars.end());
    set2map(trainChars, id2char, char2id, true);
    trainLabels.insert(evalLabels.begin(), evalLabels.end());
    set2map(trainLabels, id2label, label2id, false);
    netConf["labelSize"] = std::to_string(label2id.size());

    std::vector<Sequence> trainData;
    std::vector<Sequence> evalData;
    createData(trainRawData, word2id, char2id, label2id, trainData);
    createData(evalRawData, word2id, char2id, label2id, evalData);
    printf("=> %d / %d training and test sentences loaded.\n",
           trainData.size(), evalData.size());

    int _wordDim = std::stoi(netConf["wordDim"]);
    Eigen::MatrixXd* wordEmbedding = initializeVariable(_wordDim, word2id.size());
    if (preEmbeddingFile != "0")
        preEmbLookUp(*wordEmbedding, preEmbedding, id2word);

    int _charDim = std::stoi(netConf["charDim"]);
    Eigen::MatrixXd* charEmbedding = initializeVariable(_charDim, char2id.size());

    //
    // starts training
    //
    // initialize net
    CharBiLSTMNet charBiLSTMNet(netConf);

    int epoch = 100;
    float bestF1 = 0;
    for (int i = 0; i < epoch; i++) {
        time_t startTime = time(0);
        //
        // train
        //
        std::cout << "=> " << i << " epoch training starts..." << std::endl;

        int numThreads = std::stoi(netConf["numThreads"]);
        int rc;
        pthread_t threads[numThreads];
        // prepare thread args
        std::vector<int> index(trainData.size());
        std::iota(index.begin(), index.end(), 1);
        std::random_shuffle(index.begin(), index.end());
        std::vector<std::vector<int>> splitVec = splitVector<int>(index, numThreads);

        for(int j=0; j < numThreads; j++ ){
            ThreadArgs threadArgs;
            threadArgs.wordEmbedding = wordEmbedding;
            threadArgs.charEmbedding = charEmbedding;
            threadArgs.parameters= &charBiLSTMNet.parameters;
            threadArgs.trainData = &trainData;
            threadArgs.configuration = &netConf;
            threadArgs.index = &splitVec[j];
            rc = pthread_create(&threads[j], NULL, mainThread, (void *)&threadArgs);

            if (rc){
                std::cout << "Error:unable to create thread," << rc << std::endl;
                exit(-1);
            }
        }

        void *status;
        for(int j=0; j < numThreads; j++ ){
            rc = pthread_join(threads[j], &status);

            if (rc){
                std::cout << "Error:unable to join," << rc << std::endl;
                exit(-1);
            }
        }

        time_t trainTime = time(0) - startTime;
        std::printf("time elapsed: %d seconds (%.4f sec/sentence)\n", int(trainTime), float(trainTime) / trainData.size());

        //
        // eval
        //
        int numTags = 0;
        int numCorrectTags = 0;
        std::string evalOutPath =
                charBiLSTMNet.configuration["modelDir"]+"/eval_scores/eval_"+
                        std::to_string(i)+".bio";
        std::ofstream evalOutStream(evalOutPath);
        for (int j = 0; j < evalData.size(); ++j ) {
            Sequence input = evalData[j];
            processData(input, *wordEmbedding, *charEmbedding);

            bool isTrain = false;
            charBiLSTMNet.forward(input, isTrain);
            Eigen::MatrixXd pred = charBiLSTMNet.mlp->output;

            std::vector<int> predLabelIndex;
            Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
            for (int k = 0; k < input.labelIndex.size(); ++k) {
                for (int l = 0; l < pred.rows(); ++l) {
                    if (pred(l, k) == maxProba(k, 0))
                        predLabelIndex.push_back(l);
                }
            }
            for (int k = 0; k < input.wordIndex.size(); ++k) {
                evalOutStream << input.wordIndex[k] << " "
                              << id2label[input.labelIndex[k]] << " "
                              << id2label[predLabelIndex[k]] << std::endl;
            }
            evalOutStream << std::endl;
        }
        evalOutStream.close();

        // conll evaluation script to evaluate bio file
        std::string evalScorePath =
                charBiLSTMNet.configuration["modelDir"]+"/eval_scores/eval_"+
                        std::to_string(i)+".score";
        std::system((netConf["conllScorer"]+" < "+evalOutPath+" > "+evalScorePath).c_str());
        std::ifstream evalInStream(evalScorePath);
        std::string line;
        while (line[0] != 'a')
            std::getline(evalInStream, line);  // skip the first line

        // get floats from score output
        std::vector<float> lineValues;
        char* s = &line[0];
        for (; *s; s++) {
            if (isdigit(*s)) {
                char* pEnd;
                lineValues.push_back(strtod(s, &pEnd));
                s = pEnd;
            }
        }
        float accuracy = lineValues[0];
        float f1 = lineValues[4];

        if (f1 > bestF1) {
            std::printf("new best f1 on eval set: %.2f%%\n\n", f1);
            bestF1 = f1;
            charBiLSTMNet.saveNet(charBiLSTMNet.configuration,
                                  charBiLSTMNet.parameters,
                                  word2id, char2id, label2id,
                                  id2word, id2char, id2label,
                                  *wordEmbedding, *charEmbedding);
        } else {
            std::printf("f1 on eval set: %.2f%%\n\n", f1);
        }

//        float bestAcc = 0;
//        if (accuracy > bestAcc) {
//            std::printf("new best accuracy on eval set: %.2f%% (%d/%d)\n\n",
//                        accuracy, numCorrectTags, numTags);
//            bestAcc = accuracy;
//            charBiLSTMNet.saveNet(charBiLSTMNet.configuration,
//                                  charBiLSTMNet.parameters,
//                                  word2id, char2id, label2id,
//                                  id2word, id2char, id2label,
//                                  *wordEmbedding, *charEmbedding);
//        } else {
//            std::printf("accuracy on eval set: %.2f%% (%d/%d)\n\n",
//                        accuracy, numCorrectTags, numTags);
//        }
    }

    return 0;
}