//
// Created by Boliang Zhang on 6/13/17.
//

#include "../utils.h"
#include "seq2seq.h"
#include "loader.h"
#include <fstream>
#include <numeric>

struct ThreadArgs {
    std::vector<InputSeq2Seq>* trainData;
    Eigen::MatrixXd* srcTokenEmbedding;
    Eigen::MatrixXd* trgTokenEmbedding;
    std::vector<int>* index;
    std::map<std::string, std::string>* configuration;
    std::map<std::string, Eigen::MatrixXd*>* parameters;
};

void* mainThread(void* threadarg){
    struct ThreadArgs *args = (struct ThreadArgs *) threadarg;

    std::vector<InputSeq2Seq>* trainData = args->trainData;
    Eigen::MatrixXd* srcTokenEmbedding = args->srcTokenEmbedding;
    Eigen::MatrixXd* trgTokenEmbedding = args->trgTokenEmbedding;
    std::vector<int>* index = args->index;
    std::map<std::string, Eigen::MatrixXd*>* parameters = args->parameters;

    // initialize net
    SeqToSeq seqToSeq(*args->configuration, *parameters);

    int numSeqToReport = 500;
    std::vector<float> epoch_loss;
    for (int j = 0; j < index->size(); ++j){
        InputSeq2Seq input = (*trainData)[(*index)[j]-1];

        processData(input, *srcTokenEmbedding, *trgTokenEmbedding);
        //
        // network forward
        //
        bool isTrain = true;
        seqToSeq.forward(input, isTrain);
        Eigen::MatrixXd loss = seqToSeq.crossEntropyLoss->output;
        epoch_loss.push_back(loss.sum() / input.seqLen);

        //
        // network backward
        //
        seqToSeq.backward();

        //
        // network gradient check
        //
        seqToSeq.gradientCheck(input);

        //
        // network parameters update
        //
        seqToSeq.update();
        seqToSeq.updateEmbedding(srcTokenEmbedding,
                                 seqToSeq.diff[seqToSeq.name+"_srcTokenInput"],
                                 input.srcIndex);
        seqToSeq.updateEmbedding(trgTokenEmbedding,
                                 seqToSeq.diff[seqToSeq.name+"_trgTokenInput"],
                                 input.trgIndex);

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
    std::string tokenDim = argv[4];
    std::string tokenLSTMDim = argv[5];
    std::string learningRate = argv[6];
    std::string encoderStack = argv[7];
    std::string decoderStack = argv[8];
    std::string dropoutRate = argv[9];
    std::string numThread = argv[10];

    std::map<std::string, std::string> netConf =
            {
                    {"modelDir", modelDir},
                    {"tokenDim", tokenDim},
                    {"tokenLSTMDim", tokenLSTMDim},
                    {"learningRate", learningRate},
                    {"encoderStack", encoderStack},
                    {"decoderStack", decoderStack},
                    {"dropoutRate", dropoutRate},
                    {"numThreads", numThread},
            };

    // load raw data
    std::vector<InputSeq2Seq> trainData;
    std::vector<InputSeq2Seq> evalData;
    loadRawData(trainFile, trainData, true);
    loadRawData(evalFile, evalData, true);
    printf("=> %d / %d training and test sentences loaded.\n",
           trainData.size(), evalData.size());

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

    int epoch = 100;
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
            threadArgs.srcTokenEmbedding = srcTokenEmbedding;
            threadArgs.trgTokenEmbedding = trgTokenEmbedding;
            threadArgs.parameters= &seqToSeq.parameters;
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
        std::printf("time elapsed: %d seconds (%.4f sec/sentence)\n",
                    int(trainTime), float(trainTime) / trainData.size());

        //
        // eval
        //
        std::string evalOutPath =
                seqToSeq.configuration["modelDir"]+"/eval_scores/eval_"+
                std::to_string(i)+".bio";
        std::ofstream evalOutStream(evalOutPath);
        for (int j = 0; j < evalData.size(); ++j ) {
            InputSeq2Seq input = evalData[j];
            processData(input, *srcTokenEmbedding, *trgTokenEmbedding);

            bool isTrain = false;
            seqToSeq.forward(input, isTrain);
            Eigen::MatrixXd pred = seqToSeq.mlp->output;

            std::vector<int> predTokenIndex;
            Eigen::MatrixXd maxProba = pred.colwise().maxCoeff().transpose();
            for (int k = 0; k < pred.cols(); ++k) {
                for (int l = 0; l < pred.rows(); ++l) {
                    if (pred(l, k) == maxProba(k, 0))
                        predTokenIndex.push_back(l);
                }
            }
            for (int k = 0; k < input.srcIndex.size(); ++k) {
                evalOutStream << input.srcIndex[k] << "";
            }
            evalOutStream << " ";
            for (int k = 0; k < input.trgIndex.size(); ++k) {
                evalOutStream << input.trgIndex[k] << "";
            }
            evalOutStream << " ";
            for (int k = 0; k < predTokenIndex.size(); ++k) {
                evalOutStream << predTokenIndex[k] << "";
            }
            evalOutStream << std::endl;
        }
        evalOutStream.close();
    }

    return 0;
}