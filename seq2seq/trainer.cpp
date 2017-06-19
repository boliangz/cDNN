//
// Created by Boliang Zhang on 6/13/17.
//

#include "../utils.h"
#include "seq2seq.h"
#include "loader.h"
#include <fstream>
#include <numeric>

struct ThreadArgs {
    std::vector<Seq2SeqInput>* trainData;
    Eigen::MatrixXd* srcTokenEmbedding;
    Eigen::MatrixXd* trgTokenEmbedding;
    std::vector<int>* index;
    std::map<std::string, std::string>* configuration;
    std::map<std::string, Eigen::MatrixXd*>* parameters;
    float* epochLoss;
};

void* mainThread(void* threadarg){
    struct ThreadArgs *args = (struct ThreadArgs *) threadarg;

    std::vector<Seq2SeqInput>* trainData = args->trainData;
    Eigen::MatrixXd* srcTokenEmbedding = args->srcTokenEmbedding;
    Eigen::MatrixXd* trgTokenEmbedding = args->trgTokenEmbedding;
    std::vector<int>* index = args->index;
    std::map<std::string, Eigen::MatrixXd*>* parameters = args->parameters;
    float* epochLoss = args->epochLoss;

    // initialize net
    SeqToSeq seqToSeq(*args->configuration, *parameters);

    int numSeqToReport = 1000;
    std::vector<float> threadLoss;
    for (int j = 0; j < index->size(); ++j){
        Seq2SeqInput input = (*trainData)[(*index)[j]-1];

        processData(input, *srcTokenEmbedding, *trgTokenEmbedding, true);
        //
        // network forward
        //
        bool isTrain = true;
        seqToSeq.forward(input, isTrain);
        // compute token average loss
        Eigen::MatrixXd loss = seqToSeq.crossEntropyLoss->output;
        threadLoss.push_back(loss.sum() / input.trgLen);
        *epochLoss += loss.sum() / input.trgLen;

        //
        // network backward
        //
        seqToSeq.backward();

        //
        // network gradient check
        //
//        seqToSeq.gradientCheck(input);

        //
        // network parameters update
        //
        seqToSeq.update();
        seqToSeq.updateEmbedding(srcTokenEmbedding,
                                 seqToSeq.diff["srcTokenInput"],
                                 input.srcIndex);

        input.trgIndex.resize(input.trgOneHot.cols());
        seqToSeq.updateEmbedding(trgTokenEmbedding,
                                 seqToSeq.diff["trgTokenInput"],
                                 input.trgIndex);

        if ((j + 1) % numSeqToReport == 0){
            std::vector<float> v(threadLoss.end() - numSeqToReport, threadLoss.end());
            float average = std::accumulate( v.begin(), v.end(), 0.0)/v.size();
            std::cout << j + 1 << " sequences loss: " << average << std::endl;
        }
    }

    return 0;
}


int main(int argc, char* argv []) {
    // parse argument
    std::string trainFile = argv[1];
    std::string evalFile = argv[2];
    std::string modelDir = argv[3];
    std::string tokenDim = argv[4];
    std::string tokenLSTMDim = argv[5];
    std::string learningRate = argv[6];
    std::string encoderStack = argv[7];
    std::string decoderStack = argv[8];
    std::string dropoutRate = argv[9];
    std::string numThread = argv[10];
    std::string numEpoch = argv[11];

    std::map<std::string, std::string> netConf =
            {
                    {"modelDir", modelDir},
                    {"tokenDim", tokenDim},
                    {"tokenLSTMDim", tokenLSTMDim},
                    {"learningRate", learningRate},
                    {"encoderStack", encoderStack},
                    {"decoderStack", decoderStack},
                    {"dropoutRate", dropoutRate},
            };

    // load mapping
    std::map<std::string, int> srcToken2Id;
    std::map<int, std::string> srcId2Token;
    std::map<std::string, int> trgToken2Id;
    std::map<int, std::string> trgId2Token;
    loadMapping(
            netConf["modelDir"],
            srcToken2Id, srcId2Token,
            trgToken2Id, trgId2Token
    );
    netConf["trgTokenSize"] = std::to_string(trgToken2Id.size());

    // load raw data
    std::vector<Seq2SeqInput> trainData;
    std::vector<Seq2SeqInput> evalData;
    loadRawData(trainFile, srcToken2Id, trgToken2Id, trainData, true);
    loadRawData(evalFile, srcToken2Id, trgToken2Id, evalData, true);
    int numTrainToken = 0;
    int numEvalToken = 0;
    for (int i = 0; i < trainData.size(); ++i)
        numTrainToken += trainData[i].srcIndex.size();
    for (int i = 0; i < evalData.size(); ++i)
        numEvalToken += evalData[i].srcIndex.size();

//    trainData.resize(10);
//    evalData.resize(10);

    printf("=> %d / %d training and test sentences loaded (%d / %d tokens).\n",
           trainData.size(), evalData.size(), numTrainToken, numEvalToken);

    // initialize src and trg token embeddings
    Eigen::MatrixXd* srcTokenEmbedding = initializeVariable(
            std::stoi(tokenDim),
            srcToken2Id.size()
    );
    Eigen::MatrixXd* trgTokenEmbedding = initializeVariable(
            std::stoi(tokenDim),
            trgToken2Id.size()
    );

    //
    // starts training
    //
    // initialize net
    SeqToSeq seqToSeq(netConf);

    int epoch = std::stoi(numEpoch);
    for (int i = 0; i < epoch; i++) {
        time_t startTime = time(0);
        //
        // train
        //
        std::cout << "=> " << i << " epoch training starts..." << std::endl;

        int numThreads = std::stoi(numThread);
        int rc;
        pthread_t threads[numThreads];
        // prepare thread args
        std::vector<int> index(trainData.size());
        std::iota(index.begin(), index.end(), 1);
        std::random_shuffle(index.begin(), index.end());
        std::vector<std::vector<int>> splitVec = splitVector<int>(index, numThreads);
        float epochLoss = 0;

        for(int j=0; j < numThreads; j++ ){
            ThreadArgs threadArgs;
            threadArgs.srcTokenEmbedding = srcTokenEmbedding;
            threadArgs.trgTokenEmbedding = trgTokenEmbedding;
            threadArgs.parameters= &seqToSeq.parameters;
            threadArgs.trainData = &trainData;
            threadArgs.configuration = &netConf;
            threadArgs.index = &splitVec[j];
            threadArgs.epochLoss = &epochLoss;
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

        float avgEpochLoss = epochLoss / float(trainData.size());
        std::printf("epoch loss: %.4f\n", avgEpochLoss);
        time_t trainTime = time(0) - startTime;
        std::printf("time elapsed: %d seconds (%.4f sec/sentence)\n",
                    int(trainTime), float(trainTime) / trainData.size());

        //
        // test on eval
        //
        std::string evalOutPath =
                seqToSeq.configuration["modelDir"]+"/eval_scores/eval_"+
                std::to_string(i)+".bio";
        std::ofstream evalOutStream(evalOutPath);
        for (int j = 0; j < evalData.size(); ++j ) {
            Seq2SeqInput input = evalData[j];
            processData(input, *srcTokenEmbedding, *trgTokenEmbedding, false);

            bool isTrain = false;
            seqToSeq.forward(input, isTrain);
            Eigen::MatrixXd pred = seqToSeq.output;

            std::vector<int> predTokenIndex;
            Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
            for (int k = 0; k < pred.cols(); ++k) {
                for (int l = 0; l < pred.rows(); ++l) {
                    if (pred(l, k) == maxProba(k, 0))
                        predTokenIndex.push_back(l);
                }
            }
            for (int k = 0; k < input.srcIndex.size(); ++k) {
                evalOutStream << srcId2Token[input.srcIndex[k]] << "";
            }
            evalOutStream << " ";
            for (int k = 0; k < input.trgIndex.size(); ++k) {
                evalOutStream << trgId2Token[input.trgIndex[k]] << "";
            }
            evalOutStream << " ";
            for (int k = 0; k < predTokenIndex.size(); ++k) {
                evalOutStream << trgId2Token[predTokenIndex[k]] << "";
            }
            evalOutStream << std::endl;
        }
        evalOutStream.close();

        //
        // test on train
        //
        std::string trainOutPath =
                seqToSeq.configuration["modelDir"]+"/eval_scores/train_"+
                std::to_string(i)+".bio";
        std::ofstream trainOutStream(trainOutPath);
        for (int j = 0; j < trainData.size(); ++j ) {
            Seq2SeqInput input = trainData[j];
            processData(input, *srcTokenEmbedding, *trgTokenEmbedding, false);

            bool isTrain = false;
            seqToSeq.forward(input, isTrain);
            Eigen::MatrixXd pred = seqToSeq.output;

            std::vector<int> predTokenIndex;
            Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
            for (int k = 0; k < pred.cols(); ++k) {
                for (int l = 0; l < pred.rows(); ++l) {
                    if (pred(l, k) == maxProba(k, 0))
                        predTokenIndex.push_back(l);
                }
            }
            for (int k = 0; k < input.srcIndex.size(); ++k) {
                trainOutStream << srcId2Token[input.srcIndex[k]] << "";
            }
            trainOutStream << " ";
            for (int k = 0; k < input.trgIndex.size(); ++k) {
                trainOutStream << trgId2Token[input.trgIndex[k]] << "";
            }
            trainOutStream << " ";
            for (int k = 0; k < predTokenIndex.size(); ++k) {
                trainOutStream << trgId2Token[predTokenIndex[k]] << "";
            }
            trainOutStream << std::endl;
        }
        trainOutStream.close();
    }

    return 0;
}