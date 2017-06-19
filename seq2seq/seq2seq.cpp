//
// Created by Boliang Zhang on 6/10/17.
//
#include "seq2seq.h"

SeqToSeq::SeqToSeq(const std::map<std::string, std::string> & configuration){
    this->configuration = configuration;

    int tokenDim = std::stoi(configuration.at("tokenDim"));
    int tokenLSTMDim = std::stoi(configuration.at("tokenLSTMDim"));
    int encoderStack = std::stoi(configuration.at("encoderStack"));
    int decoderStack = std::stoi(configuration.at("decoderStack"));
    float dropoutRate = std::stof(configuration.at("dropoutRate"));
    int trgTokenSize = std::stoi(configuration.at("trgTokenSize"));

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

    // initialize loss
    crossEntropyLoss = new CrossEntropyLoss("crossEntropyLoss");
}

SeqToSeq::SeqToSeq(const std::map<std::string, std::string> & configuration,
                   const std::map<std::string, Eigen::MatrixXd*>& parameters) {
    this->configuration = configuration;

    int tokenDim = std::stoi(configuration.at("tokenDim"));
    int tokenLSTMDim = std::stoi(configuration.at("tokenLSTMDim"));
    int encoderStack = std::stoi(configuration.at("encoderStack"));
    int decoderStack = std::stoi(configuration.at("decoderStack"));
    float dropoutRate = std::stof(configuration.at("dropoutRate"));
    int trgTokenSize = std::stoi(configuration.at("trgTokenSize"));

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

    // initialize loss
    crossEntropyLoss = new CrossEntropyLoss("crossEntropyLoss");
}


void SeqToSeq::forward(const Input & input, bool isTrain) {
    const Seq2SeqInput& seq2SeqInput = static_cast<const Seq2SeqInput&>(input);

    // stacked encoder forward
    Eigen::MatrixXd lstmInput = seq2SeqInput.srcEmb;
    for (int i = 0; i < encoder.size(); i++){
        encoder[i]->forward(lstmInput, NULL, NULL);
        lstmInput = encoder[i]->output;
    }

    // forward when training a model
    if (isTrain) {
        // stacked decoder forward
        lstmInput = seq2SeqInput.trgEmb;
        for (int i = 0; i < decoder.size(); i++){
            if (i == 0){
                Eigen::MatrixXd h_prev = encoder.back()->
                        cache[encoder.back()->name+"_h"].rightCols(1);
                Eigen::MatrixXd c_prev = encoder.back()->
                        cache[encoder.back()->name+"_c"].rightCols(1);
                decoder[i]->forward(lstmInput, &h_prev, &c_prev);
            }
            else
                decoder[i]->forward(lstmInput, NULL, NULL);
            lstmInput = decoder[i]->output;
        }

        // mlp forward
        mlp->forward(decoder.back()->output);

        // cross entropy loss forward
        crossEntropyLoss->forward(mlp->output, seq2SeqInput.trgOneHot);

        output = crossEntropyLoss->output;
    }
        // forward when testing
    else {
        int tokenLSTMDim = std::stoi(configuration.at("tokenLSTMDim"));
        int trgTokenSize = std::stoi(this->configuration["trgTokenSize"]);

        std::vector<Eigen::MatrixXd> pred;

        // init <s> embeddings as the start of the predicted sequence
        Eigen::MatrixXd start = seq2SeqInput.trgEmb.col(0);

        // init h_prev and c_prev
        Eigen::MatrixXd h_prev(tokenLSTMDim, decoder.size());
        h_prev.fill(0);
        Eigen::MatrixXd c_prev(tokenLSTMDim, decoder.size());
        c_prev.fill(0);

        h_prev.col(0) = encoder.back()->
                cache[encoder.back()->name+"_h"].rightCols(1);
        c_prev.col(0) = encoder.back()->
                cache[encoder.back()->name+"_c"].rightCols(1);

        // starts from <s>, generate each token until </s> met
        lstmInput = start;
        while (pred.size() < 100) {
            // stacked decoder forward
            for (int i = 0; i < decoder.size(); i++){
                Eigen::MatrixXd h_p = h_prev.col(i);
                Eigen::MatrixXd c_p = c_prev.col(i);
                decoder[i]->forward(lstmInput, &h_p, &c_p);
                lstmInput = decoder[i]->output;

                // update h_prev and c_prev
                h_prev.col(i) = decoder[i]->cache[decoder[i]->name+"_h"];
                c_prev.col(i) = decoder[i]->cache[decoder[i]->name+"_c"];
            }

            // mlp forward
            mlp->forward(decoder.back()->output);

            // push to prediction
            pred.push_back(mlp->output);

            // get the token index of the predicted token
            int tokenIndex;
            for (int i = 0; i < mlp->output.rows(); ++i)
                if (mlp->output(i, 0) == mlp->output.maxCoeff()){
                    tokenIndex = i;
                    break;
                }

            // break if the predicted token is </s>
            if (tokenIndex == 1)
                break;

            // update lstmInput with the token embedding of the current decoder
            // output
            lstmInput = seq2SeqInput.trgTokenEmb->col(tokenIndex);
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
    int tokenLSTMDim = std::stoi(configuration["tokenLSTMDim"]);
    Eigen::MatrixXd dy = mlp->inputDiff;
    for (long i = decoder.size(); i --> 0;){
        decoder[i]->backward(dy, NULL, NULL);
        dy = decoder[i]->inputDiff;
        diff.insert(decoder[i]->diff.begin(),
                    decoder[i]->diff.end());
    }
    diff["trgTokenInput"] = decoder[0]->inputDiff;

    // stacked encoder backward
    int srcLen = encoder[0]->cache[encoder[0]->name+"_input"].cols();
    dy = Eigen::MatrixXd::Zero(tokenLSTMDim, srcLen);
    for(long i = encoder.size(); i--> 0;) {
        if (i == encoder.size()-1){
            Eigen::MatrixXd dh_next = decoder[0]->diff[decoder[0]->name+"_h_next"].col(0);
            Eigen::MatrixXd dc_next = decoder[0]->diff[decoder[0]->name+"_c_next"].col(0);
            encoder[i]->backward(dy, &dh_next, &dc_next);
        } else {
            encoder[i]->backward(dy, NULL, NULL);
        }

        dy = encoder[i]->inputDiff;
        diff.insert(encoder[i]->diff.begin(),
                    encoder[i]->diff.end());
    }
    diff["srcTokenInput"] = encoder[0]->inputDiff;
}

void SeqToSeq::gradientCheck(Seq2SeqInput & input) {
    std::map<std::string, Eigen::MatrixXd*> additionalParam;
    additionalParam["srcTokenInput"] = &input.srcEmb;
    additionalParam["trgTokenInput"] = &input.trgEmb;

    Net::gradientCheck(input, additionalParam);
}

void SeqToSeq::update() {
    float learningRate = std::stof(configuration["learningRate"]);
    Layer::update(learningRate);
}