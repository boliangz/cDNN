//
// Created by Boliang Zhang on 6/10/17.
//
#include "seq2seq.h"

SeqToSeq::SeqToSeq(const std::map<std::string, std::string> & configuration){
    int tokenDim = std::stoi(configuration.at("tokenDim"));
    int tokenLSTMDim = std::stoi(configuration.at("tokenLSTMDim"));
    int encoderStack = std::stoi(configuration.at("encoderStack"));
    int decoderStack = std::stoi(configuration.at("decoderStack"));
    float dropoutRate = std::stof(configuration.at("dropoutRate"));
    int trgTokenSize = std::stoi(this->configuration["trgTokenSize"]);

    // initialize encoder
    for (int i = 0; i < encoderStack; ++i) {
        LSTM *encoderLSTM;
        if (i == 0)
            encoderLSTM = new LSTM(tokenDim,
                                   tokenLSTMDim,
                                   "encoderLSTM"+std::to_string(i));
        else
            encoderLSTM = new LSTM(tokenLSTMDim,
                                   tokenLSTMDim,
                                   "encoderLSTM"+std::to_string(i));
        parameters.insert(encoderLSTM->parameters.begin(),
                          encoderLSTM->parameters.end());
        encoder.push_back(encoderLSTM);
    }
    // initialize decoder
    for (int i = 0; i < decoderStack; ++i) {
        LSTM *decoderLSTM;
        if (i == 0)
            decoderLSTM = new LSTM(tokenDim,
                                   tokenLSTMDim,
                                   "decoderLSTM"+std::to_string(i));
        else
            decoderLSTM = new LSTM(tokenLSTMDim,
                                   tokenLSTMDim,
                                   "decoderLSTM"+std::to_string(i));
        parameters.insert(decoderLSTM->parameters.begin(),
                          decoderLSTM->parameters.end());
        decoder.push_back(decoderLSTM);
    }

    // initialize mlp
    mlp = new MLP(tokenLSTMDim, trgTokenSize, "mlp");
    parameters.insert(mlp->parameters.begin(),
                      mlp->parameters.end());
}

SeqToSeq::SeqToSeq(const std::map<std::string, std::string> & configuration,
                   const std::map<std::string, Eigen::MatrixXd*>& parameters) {
    int tokenDim = std::stoi(configuration.at("tokenDim"));
    int tokenLSTMDim = std::stoi(configuration.at("tokenLSTMDim"));
    int encoderStack = std::stoi(configuration.at("encoderStack"));
    int decoderStack = std::stoi(configuration.at("decoderStack"));
    float dropoutRate = std::stof(configuration.at("dropoutRate"));
    int trgTokenSize = std::stoi(this->configuration["trgTokenSize"]);

    // initialize encoder
    for (int i = 0; i < encoderStack; ++i) {
        LSTM *encoderLSTM;
        if (i == 0)
            encoderLSTM = new LSTM(tokenDim,
                                   tokenLSTMDim,
                                   "encoderLSTM"+std::to_string(i),
                                   parameters);
        else
            encoderLSTM = new LSTM(tokenLSTMDim,
                                   tokenLSTMDim,
                                   "encoderLSTM"+std::to_string(i),
                                   parameters);
        this->parameters.insert(encoderLSTM->parameters.begin(),
                                encoderLSTM->parameters.end());
        encoder.push_back(encoderLSTM);
    }
    // initialize decoder
    for (int i = 0; i < decoderStack; ++i) {
        LSTM *decoderLSTM;
        if (i == 0)
            decoderLSTM = new LSTM(tokenDim,
                                   tokenLSTMDim,
                                   "decoderLSTM"+std::to_string(i),
                                   parameters);
        else
            decoderLSTM = new LSTM(tokenLSTMDim,
                                   tokenLSTMDim,
                                   "decoderLSTM"+std::to_string(i),
                                   parameters);
        this->parameters.insert(decoderLSTM->parameters.begin(),
                                decoderLSTM->parameters.end());
        decoder.push_back(decoderLSTM);
    }

    // initialize mlp
    mlp = new MLP(tokenLSTMDim, trgTokenSize, "mlp", parameters);
    this->parameters.insert(mlp->parameters.begin(),
                            mlp->parameters.end());
}


void SeqToSeq::forward(const InputSeq2Seq & input, bool isTrain) {
    // stacked encoder forward
    Eigen::MatrixXd lstmInput = input.srcEmb;
    for (int i = 0; i < encoder.size(); i++){
        encoder[i]->forward(lstmInput, NULL, NULL);
        lstmInput = encoder[i]->output;
    }

    // forward when training a model
    if (isTrain) {
        // stacked decoder forward
        lstmInput = input.trgEmb;
        for (int i = 0; i < decoder.size(); i++){
            if (i == 0){
                Eigen::MatrixXd h_prev = encoder.back()->
                        cache[encoder.back()->name+"_h"].col(input.seqLen-1);
                Eigen::MatrixXd c_prev = encoder.back()->
                        cache[encoder.back()->name+"_c"].col(input.seqLen-1);
                decoder[i]->forward(lstmInput, &h_prev, &c_prev);
            }
            else
                decoder[i]->forward(lstmInput, NULL, NULL);
            lstmInput = decoder[i]->output;
        }

        // mlp forward
        mlp->forward(decoder.back()->output);

        // cross entropy loss forward
        crossEntropyLoss->forward(mlp->output, input.trgOneHot);

        output = crossEntropyLoss->output;
    }
    // forward when testing
    else {
        int tokenDim = std::stoi(configuration.at("tokenDim"));
        int tokenLSTMDim = std::stoi(configuration.at("tokenLSTMDim"));
        int trgTokenSize = std::stoi(this->configuration["trgTokenSize"]);

        std::vector<Eigen::MatrixXd> pred;

        // init <s>
        Eigen::MatrixXd start(tokenDim, 1);
        start.fill(0);
        start(1, 0) = 1;

        // init h_prev and c_prev
        Eigen::MatrixXd h_prev(tokenLSTMDim, decoder.size());
        h_prev.fill(0);
        Eigen::MatrixXd c_prev(tokenLSTMDim, decoder.size());
        c_prev.fill(0);

        h_prev.col(0) = encoder.back()->
                cache[encoder.back()->name+"_h"].col(input.seqLen-1);
        c_prev.col(0) = encoder.back()->
                cache[encoder.back()->name+"_c"].col(input.seqLen-1);

        // starts from <s>, generate each token until </s> met
        while (true) {
            // stacked decoder forward
            lstmInput = start;
            for (int i = 0; i < decoder.size(); i++){
                Eigen::MatrixXd h_p = h_prev.col(i);
                Eigen::MatrixXd c_p = c_prev.col(i);
                if (i == 0){

                    decoder[i]->forward(lstmInput,
                                        &h_p,
                                        &c_p);
                }
                else
                    decoder[i]->forward(lstmInput,
                                        &h_p,
                                        &c_p);
                lstmInput = decoder[i]->output;

                // update h_prev and c_prev
                h_prev.col(i) = decoder[i]->cache[decoder[i]->name+"_h"];
                c_prev.col(i) = decoder[i]->cache[decoder[i]->name+"_c"];
            }

            // mlp forward
            mlp->forward(decoder.back()->output);

            // push to prediction
            pred.push_back(mlp->output);

            // break if the predicted token is </s>
            if (mlp->output(0, 0) == 1)
                break;
        }

        // prepare output
        Eigen::MatrixXd predOutput(trgTokenSize, pred.size());
        for (int i = 0; i < pred.size(); i++)
            predOutput.col(i) = pred[i];
        output = predOutput;
    }

}

void SeqToSeq::backward() {
    // clear diff
    diff.clear();

    // cross entropy backward
    crossEntropyLoss->backward();

    // mlp backward
    mlp->backward(crossEntropyLoss->inputDiff);
    diff.insert(mlp->diff.begin(),
                mlp->diff.end());

    // stacked decoder backward
    int sequenceLen = crossEntropyLoss->output.cols();
    int tokenLSTMDim = std::stoi(configuration["tokenLSTMDim"]);
    Eigen::MatrixXd dy = mlp->inputDiff;
    for (long i = decoder.size(); i --> 0;){
        decoder[i]->backward(dy, NULL, NULL);
        dy = decoder[i]->inputDiff;
        diff.insert(decoder[i]->diff.begin(),
                    decoder[i]->diff.end());
    }
    diff[name+"_trgTokenInput"] = decoder[0]->inputDiff;

    // stacked encoder backward
    dy = Eigen::MatrixXd::Zero(tokenLSTMDim, sequenceLen);
    for(long i = encoder.size(); i--> 0;) {
        if (i == encoder.size()-1){
            Eigen::MatrixXd dh_next = decoder[0]->diff[decoder[0]->name+"_h_prev"].col(0);
            Eigen::MatrixXd dc_next = decoder[0]->diff[decoder[0]->name+"_c_prev"].col(0);
            encoder[i]->backward(dy, &dh_next, &dc_next);
        } else {
            encoder[i]->backward(dy, NULL, NULL);
        }

        dy = encoder[i]->inputDiff;
        diff.insert(encoder[i]->diff.begin(),
                    encoder[i]->diff.end());
    }
    diff[name+"_srcTokenInput"] = encoder[0]->inputDiff;
}

void SeqToSeq::gradientCheck(InputSeq2Seq & input) {
    std::map<std::string,
            Eigen::MatrixXd*> additionalParam;
    additionalParam[name+"_srcTokenInput"] = &input.srcEmb;
    additionalParam[name+"_trgTokenInput"] = &input.trgEmb;

    Net::gradientCheck(input, additionalParam);
}

void SeqToSeq::update() {
    float learningRate = std::stof(configuration["learningRate"]);
    Layer::update(learningRate);
}