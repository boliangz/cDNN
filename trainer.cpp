#include "charBiLSTMNet.h"
#include "utils.h"
#include "loader.h"
#include <iostream>
#include <numeric>
#include <Eigen/Core>

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

    int numSeqToReport = 1000;
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


template<typename T>
std::vector<std::vector<T>> splitVector(const std::vector<T>& vec, size_t n)
{
    std::vector<std::vector<T>> outVec;

    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;

    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < std::min(n, vec.size()); ++i)
    {
        end += (remain > 0) ? (length + !!(remain--)) : length;

        outVec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));

        begin = end;
    }

    return outVec;
}

int main(int argc, char* argv []) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " train_bio eval_bio pretrain_emb" << std::endl;
        return 1;
    }

    std::string trainFile = argv[1];
    std::string evalFile = argv[2];
    std::string preEmbeddingFile = argv[3];

    std::map<std::string, std::string> netConf =
            {
                    {"wordDim", "50"},
                    {"charDim", "25"},
                    {"wordLSTMHiddenDim", "100"},
                    {"charLSTMHiddenDim", "25"},
                    {"learningRate", "0.01"},
                    {"dropoutRate", "0.5"},
                    {"preEmb", preEmbeddingFile},
                    {"numThreads", "2"},
                    {"modelDir", "/Users/boliangzhang/Documents/Phd/cDNN/model"}
            };

    RAWDATA trainRawData;
    RAWDATA evalRawData;
    loadRawData(trainFile, trainRawData);
    loadRawData(evalFile, evalRawData);

    std::set<std::string> trainWords, trainLabels, trainChars,
            evalWords, evalLabels, evalChars;
    createTokenSet(trainRawData, trainWords, trainLabels, trainChars);
    createTokenSet(evalRawData, evalWords, evalLabels, evalChars);

    std::map<std::string, Eigen::MatrixXd> preEmbedding;
    std::printf("loading pre-trained embedding from: %s \n",
                preEmbeddingFile.c_str());
    loadPreEmbedding(preEmbeddingFile, preEmbedding);

    expandWordSet(trainWords, evalWords, preEmbedding);

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
    trainData.resize(100);

    int wordDim = std::stoi(netConf["wordDim"]);
    Eigen::MatrixXd* wordEmbedding = initializeVariable(wordDim, word2id.size());
    preEmbLookUp(*wordEmbedding, preEmbedding, id2word);

    int charDim = std::stoi(netConf["charDim"]);
    Eigen::MatrixXd* charEmbedding = initializeVariable(charDim, char2id.size());

    //
    // starts training
    //
    // initialize net
    CharBiLSTMNet charBiLSTMNet(netConf);

    int epoch = 100;
    float bestAcc = 0;
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
        std::vector<std::vector<int>> splitVec = splitVector(index, numThreads);

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

//        float average = std::accumulate( epoch_loss.begin(), epoch_loss.end(), 0.0)/epoch_loss.size();
//        std::cout << "epoch loss: " << average << std::endl;
        time_t trainTime = time(0) - startTime;
        std::printf("time elapsed: %d seconds (%.4f sec/sentence)\n", int(trainTime), float(trainTime) / trainData.size());

        //
        // eval
        //
        int numTags = 0;
        int numCorrectTags = 0;
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
            numTags += input.labelIndex.size();
            for (int k = 0; k < input.labelIndex.size(); ++k) {
                if (predLabelIndex[k] == input.labelIndex[k])
                    numCorrectTags += 1;
            }
        }
        float acc = float(numCorrectTags) / numTags * 100;
        if (acc > bestAcc) {
            std::printf("new best accuracy on eval set: %.2f%% (%d/%d)\n\n",
                        acc, numCorrectTags, numTags);
            bestAcc = acc;
            charBiLSTMNet.saveNet(charBiLSTMNet.configuration,
                                  charBiLSTMNet.parameters,
                                  word2id, char2id, label2id,
                                  id2word, id2char, id2label,
                                  *wordEmbedding, *charEmbedding);
        } else {
            std::printf("accuracy on eval set: %.2f%% (%d/%d)\n\n",
                        acc, numCorrectTags, numTags);
        }
    }

    return 0;
}