//
// Created by Boliang Zhang on 5/19/17.
//
#include "nn.h"
#include "utils.h"
#include "seqLabeling/loader.h"
#include <vector>
#include <random>
#include <Eigen/Core>
#include <iostream>

//
// Base class implementation
//
void Layer::copyParameters(const std::map<std::string, Eigen::MatrixXd*>& parameters){
    std::map<std::string, Eigen::MatrixXd*>::iterator it;
    for (it = this->parameters.begin(); it != this->parameters.end(); ++it) {
        std::string param_name = it->first;
        Eigen::MatrixXd* param = it->second;

        delete param;

        it->second = parameters.at(param_name);
    }
}

void Layer::forward(const std::vector<Eigen::MatrixXd> & input) {
    batchCache.clear();
    batchOutput.clear();
    for (int i = 0; i < input.size(); ++i) {
        forward(input[i]);
        batchCache.push_back(cache);
        batchOutput.push_back(cache[name+"_output"]);
    }
}

void Layer::backward(const std::vector<Eigen::MatrixXd> & dy) {
    batchDiff.clear();
    batchInputDiff.clear();
    for (int i = 0; i < dy.size(); ++i) {
        cache = batchCache[i];
        backward(dy[i]);

        batchDiff.push_back(diff);
        batchInputDiff.push_back(diff[name+"_input"]);
    }
    for (int i = 0; i < batchDiff.size(); ++i){
        for (auto & d: batchDiff[i]) {
            if (d.first == name+"_input") continue;
            if (i == 0){
                diff[d.first] = d.second;
            } else {
                diff[d.first] = diff[d.first].array() + d.second.array();
            }
        }
    }
}

void Layer::update(float learningRate) {
    mtx.lock();

    std::map<std::string, Eigen::MatrixXd*>::iterator it;
    for (it = parameters.begin(); it != parameters.end(); ++it) {
        auto param_name = it->first;
        auto param = it->second;

        *param -= learningRate * diff[param_name];
    }

    mtx.unlock();
}

void Layer::gradientCheck() {
    auto input = cache[name+"_input"];
    auto dy = cache[name+"_dy"];

    std::cout << "=> " + name + " gradient checking..." << std::endl;

    int num_checks = 10;
    double delta = 10e-5;

    std::vector<std::pair<std::string, Eigen::MatrixXd*> > paramToCheck;
    std::map<std::string, Eigen::MatrixXd*>::iterator it;
    for (it = parameters.begin(); it != parameters.end(); ++it)
        paramToCheck.push_back({it->first, it->second});
    paramToCheck.push_back({name+"_input", &input});

    std::cout.precision(15);
    for (int i = 0; i < paramToCheck.size(); ++i) {
        auto paramName = paramToCheck[i].first;
        auto param = paramToCheck[i].second;

        std::printf("checking %s %s\n", name.c_str(), paramName.c_str());

        assert(param->rows() == diff[paramName].rows()
               || param->cols() == diff[paramName].cols());

        for (int j = 0; j < num_checks; ++j) {
            int randRow = rand() % (int)(param->rows());
            int randCol = rand() % (int)(param->cols());

            double originalVal = (*param)(randRow, randCol);

            (*param)(randRow, randCol) = originalVal - delta;
            forward(input);
            auto output0 = cache[name+"_output"];

            (*param)(randRow, randCol) = originalVal + delta;
            forward(input);
            auto output1 = cache[name+"_output"];

            (*param)(randRow, randCol) = originalVal;

            double analyticGrad = diff[paramName](randRow, randCol);
            double numericalGrad;
            if (dy.data() == NULL)
                numericalGrad = (output1 - output0).array().sum()
                                / (2.0 * delta);
            else
                numericalGrad =
                        ((output1 - output0).array() * dy.array()).sum()
                        / (2.0 * delta);
            double rel_error = fabs(analyticGrad - numericalGrad)
                               / fabs(analyticGrad + numericalGrad);

            if (rel_error > 10e-5)
                std::cout << "\t"
                          << numericalGrad << ", "
                          << analyticGrad << " ==> "
                          << rel_error << std::endl;
        }
    }
}


//
// MLP implementation
//
MLP::MLP(int inputSize, int hiddenDim, std::string name) {
    this->name = name;

    parameters[name + "_W"] = initializeVariable(inputSize, hiddenDim);
    parameters[name + "_b"] = initializeVariable(hiddenDim, 1);
}


MLP::MLP(int inputSize, int hiddenDim, std::string name,
         const std::map<std::string, Eigen::MatrixXd*>& parameters):
        MLP(inputSize, hiddenDim, name){
    Layer::copyParameters(parameters);
}

void MLP::forward(const Eigen::MatrixXd & input){
    auto W = parameters[name + "_W"];
    auto b = parameters[name + "_b"];

    Eigen::MatrixXd h_inner = (input.transpose() * (*W)).transpose().colwise() + (*b).col(0);

    Eigen::MatrixXd output = softmax(h_inner);

    cache[name + "_input"] = input;
    cache[name + "_output"] = output;
    this->output = output;
}

void MLP::backward(const Eigen::MatrixXd & dy){
    cache[name+"_dy"] = dy;

    auto W = parameters[name + "_W"];

    auto input = cache[name + "_input"];
    auto output = cache[name + "_output"];

    long sequenceLen = dy.cols();
    long hiddenDim = W->cols();
    long inputSize = W->rows();

    std::vector<Eigen::MatrixXd> tmp = dsoftmax(output);

    Eigen::MatrixXd dh(hiddenDim, sequenceLen);

    for (int i = 0; i < sequenceLen; i++) {
        tmp[i] = tmp[i].array().colwise() * dy.col(i).array();
        dh.col(i) = tmp[i].colwise().sum().transpose();
    }

    diff[name + "_W"] = Eigen::MatrixXd::Zero(inputSize, hiddenDim);
    diff[name + "_b"] = Eigen::MatrixXd::Zero(hiddenDim, 1);
    diff[name + "_input"] = Eigen::MatrixXd::Zero(inputSize, sequenceLen);


    for (int i = 0; i < sequenceLen; i++) {
        Eigen::MatrixXd dW = input.col(i) * dh.col(i).transpose();
        Eigen::MatrixXd db = dh.col(i);
        Eigen::MatrixXd dx = (*W) * dh.col(i);

        diff[name + "_W"] += dW;
        diff[name + "_b"] += db;
        diff[name + "_input"].col(i) = dx;
    }

    inputDiff = diff[name+"_input"];
}


//
// LSTM implementation
//
LSTM::LSTM(int inputSize, int hiddenDim, std::string name){
    this->name = name;

    parameters[name + "_Wi"] = initializeVariable(inputSize + hiddenDim, hiddenDim);
    parameters[name + "_Wf"] = initializeVariable(inputSize + hiddenDim, hiddenDim);
    parameters[name + "_Wc"] = initializeVariable(inputSize + hiddenDim, hiddenDim);
    parameters[name + "_Wo"] = initializeVariable(inputSize + hiddenDim, hiddenDim);
    parameters[name + "_bi"] = initializeVariable(hiddenDim, 1);
    parameters[name + "_bf"] = initializeVariable(hiddenDim, 1);
    parameters[name + "_bc"] = initializeVariable(hiddenDim, 1);
    parameters[name + "_bo"] = initializeVariable(hiddenDim, 1);
}

LSTM::LSTM(int inputSize, int hiddenDim, std::string name, bool isBatch):
        LSTM(inputSize, hiddenDim, name){
    this->isBatch = isBatch;
}

LSTM::LSTM(int inputSize, int hiddenDim, std::string name,
           const std::map<std::string, Eigen::MatrixXd*>& parameters):
        LSTM::LSTM(inputSize, hiddenDim, name){
     copyParameters(parameters);
}

LSTM::LSTM(int inputSize, int hiddenDim, std::string name, bool isBatch,
           const std::map<std::string, Eigen::MatrixXd*>& parameters):
        LSTM::LSTM(inputSize, hiddenDim, name, isBatch){
    copyParameters(parameters);
}

void LSTM::forward(const Eigen::MatrixXd & input){
    auto Wi = parameters[name + "_Wi"];
    auto Wf = parameters[name + "_Wf"];
    auto Wc = parameters[name + "_Wc"];
    auto Wo = parameters[name + "_Wo"];
    auto bi = parameters[name + "_bi"];
    auto bf = parameters[name + "_bf"];
    auto bc = parameters[name + "_bc"];
    auto bo = parameters[name + "_bo"];

    long sequenceLen = input.cols();
    long hiddenDim = Wi->cols();
    long inputSize = Wi->rows() - hiddenDim;

    Eigen::MatrixXd h_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);
    Eigen::MatrixXd c_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);

    cache[name + "_input"] = input;
    cache[name + "_h"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    cache[name + "_c"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    cache[name + "_hi"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    cache[name + "_hf"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    cache[name + "_ho"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    cache[name + "_hc"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);
    cache[name + "_output"] = Eigen::MatrixXd::Zero(hiddenDim, sequenceLen);

    for(int i=0; i < sequenceLen; i++){
        Eigen::MatrixXd z(hiddenDim + inputSize, 1);
        z << h_prev, input.col(i);

        Eigen::MatrixXd hf_ = (z.transpose() * (*Wf)).transpose() + (*bf);
        Eigen::MatrixXd hf = sigmoid(hf_);
        Eigen::MatrixXd hi_ = (z.transpose() * (*Wi)).transpose() + (*bi);
        Eigen::MatrixXd hi = sigmoid(hi_);
        Eigen::MatrixXd ho_ = (z.transpose() * (*Wo)).transpose() + (*bo);
        Eigen::MatrixXd ho = sigmoid(ho_);
        Eigen::MatrixXd hc_ = (z.transpose() * (*Wc)).transpose() + (*bc);
        Eigen::MatrixXd hc = tanh(hc_);
        Eigen::MatrixXd c = hf.array() * c_prev.array() + hi.array() * hc.array();
        Eigen::MatrixXd h = ho.array() * tanh(c).array();

        cache[name + "_h"].col(i) = h;
        cache[name + "_c"].col(i) = c;
        cache[name + "_hi"].col(i) = hi;
        cache[name + "_hf"].col(i) = hf;
        cache[name + "_ho"].col(i) = ho;
        cache[name + "_hc"].col(i) = hc;
        cache[name + "_output"].col(i) = h;

        h_prev = h;
        c_prev = c;
    }
    this->output = cache[name+"_output"];
}

void LSTM::backward(const Eigen::MatrixXd & dy) {
    cache[name+"_dy"] = dy;

    auto Wi = parameters[name + "_Wi"];
    auto Wf = parameters[name + "_Wf"];
    auto Wc = parameters[name + "_Wc"];
    auto Wo = parameters[name + "_Wo"];

    long sequenceLen = dy.cols();
    long hiddenDim = Wi->cols();
    long inputSize = Wi->rows() - hiddenDim;

    Eigen::MatrixXd dh_next = Eigen::MatrixXd::Zero(hiddenDim, 1);
    Eigen::MatrixXd dc_next = Eigen::MatrixXd::Zero(hiddenDim, 1);

    // initialize parameter diff to zero.
    diff[name + "_Wi"] = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    diff[name + "_Wf"] = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    diff[name + "_Wc"] = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    diff[name + "_Wo"] = Eigen::MatrixXd::Zero(inputSize + hiddenDim, hiddenDim);
    diff[name + "_bi"] = Eigen::MatrixXd::Zero(hiddenDim, 1);
    diff[name + "_bf"] = Eigen::MatrixXd::Zero(hiddenDim, 1);
    diff[name + "_bc"] = Eigen::MatrixXd::Zero(hiddenDim, 1);
    diff[name + "_bo"] = Eigen::MatrixXd::Zero(hiddenDim, 1);
    diff[name + "_input"] = Eigen::MatrixXd::Zero(inputSize, sequenceLen);

    for(long t = sequenceLen; t --> 0;) {
        Eigen::MatrixXd x_t = cache[name + "_input"].col(t);
        Eigen::MatrixXd hi_t = cache[name + "_hi"].col(t);
        Eigen::MatrixXd hf_t = cache[name + "_hf"].col(t);
        Eigen::MatrixXd hc_t = cache[name + "_hc"].col(t);
        Eigen::MatrixXd ho_t = cache[name + "_ho"].col(t);
        Eigen::MatrixXd h_t = cache[name + "_h"].col(t);
        Eigen::MatrixXd c_t = cache[name + "_c"].col(t);
        Eigen::MatrixXd dy_t = dy.col(t);

        Eigen::MatrixXd c_prev;
        Eigen::MatrixXd h_prev;
        if (t == 0){
            c_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);
            h_prev = Eigen::MatrixXd::Zero(hiddenDim, 1);
        }
        else {
            c_prev = cache[name + "_c"].col(t-1);
            h_prev = cache[name + "_h"].col(t-1);
        }

        Eigen::MatrixXd z(hiddenDim + inputSize, 1);
        z << h_prev, x_t;

        Eigen::MatrixXd dh = dy_t + dh_next;

        Eigen::MatrixXd tanhCt = tanh(c_t);
        Eigen::MatrixXd dc = ho_t.array() * dtanh(tanhCt).array() * dh.array() + dc_next.array();
        Eigen::MatrixXd dho = tanh(c_t).array() * dh.array();
        Eigen::MatrixXd dho_inner = dsigmoid(ho_t).array() * dho.array();
        Eigen::MatrixXd dhf = c_prev.array() * dc.array();
        Eigen::MatrixXd dhf_inner = dsigmoid(hf_t).array() * dhf.array();
        Eigen::MatrixXd dhi = hc_t.array() * dc.array();
        Eigen::MatrixXd dhi_inner = dsigmoid(hi_t).array() * dhi.array();
        Eigen::MatrixXd dhc = hi_t.array() * dc.array();
        Eigen::MatrixXd dhc_inner = dtanh(hc_t).array() * dhc.array();

        Eigen::MatrixXd dWc = z * dhc_inner.transpose();
        Eigen::MatrixXd dbc = dhc_inner;

        Eigen::MatrixXd dWi = z * dhi_inner.transpose();
        Eigen::MatrixXd dbi = dhi_inner;

        Eigen::MatrixXd dWf = z * dhf_inner.transpose();
        Eigen::MatrixXd dbf = dhf_inner;

        Eigen::MatrixXd dWo = z * dho_inner.transpose();
        Eigen::MatrixXd dbo = dho_inner;

        Eigen::MatrixXd dz = (*Wi) * dhi_inner
                             + (*Wc) * dhc_inner
                             + (*Wo) * dho_inner
                             + (*Wf) * dhf_inner;
        Eigen::MatrixXd dx = dz.block(hiddenDim, 0, inputSize, 1);

        // update dh_next and dc_next
        dh_next = dz.block(0, 0, hiddenDim, 1);
        dc_next = hf_t.array() * dc.array();

        diff[name + "_Wi"] += dWi;
        diff[name + "_Wf"] += dWf;
        diff[name + "_Wc"] += dWc;
        diff[name + "_Wo"] += dWo;
        diff[name + "_bi"] += dbi;
        diff[name + "_bf"] += dbf;
        diff[name + "_bc"] += dbc;
        diff[name + "_bo"] += dbo;
        diff[name + "_input"].col(t) += dx;
    }
    this->inputDiff = diff[name+"_input"];
}

//
// Bi-LSTM implementation
//
BiLSTM::BiLSTM(int inputSize, int hiddenDim, std::string name){
    this->name = name;
    fwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_fwdLSTM");
    bwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_bwdLSTM");

    for ( const auto& p: fwdLSTM->parameters ){
        parameters[p.first] = p.second;
    }
    parameters.insert(fwdLSTM->parameters.begin(),
                      fwdLSTM->parameters.end());
    parameters.insert(bwdLSTM->parameters.begin(),
                      bwdLSTM->parameters.end());}

BiLSTM::BiLSTM(int inputSize, int hiddenDim, std::string name, bool isBatch){
    this->name = name;
    fwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_fwdLSTM", isBatch);
    bwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_bwdLSTM", isBatch);

    parameters.insert(fwdLSTM->parameters.begin(),
                      fwdLSTM->parameters.end());
    parameters.insert(bwdLSTM->parameters.begin(),
                      bwdLSTM->parameters.end());}

BiLSTM::BiLSTM(int inputSize, int hiddenDim, std::string name,
               const std::map<std::string, Eigen::MatrixXd*>& parameters){
    this->name = name;
    fwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_fwdLSTM", parameters);
    bwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_bwdLSTM", parameters);

    for ( const auto& p: fwdLSTM->parameters ){
        this->parameters[p.first] = p.second;
    }
    this->parameters.insert(fwdLSTM->parameters.begin(),
                            fwdLSTM->parameters.end());
    this->parameters.insert(bwdLSTM->parameters.begin(),
                            bwdLSTM->parameters.end());
}

BiLSTM::BiLSTM(int inputSize, int hiddenDim, std::string name, bool isBatch,
               const std::map<std::string, Eigen::MatrixXd*>& parameters){
    this->name = name;
    fwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_fwdLSTM",
                       isBatch, parameters);
    bwdLSTM = new LSTM(inputSize, hiddenDim, this->name + "_bwdLSTM",
                       isBatch, parameters);

    this->parameters.insert(fwdLSTM->parameters.begin(),
                            fwdLSTM->parameters.end());
    this->parameters.insert(bwdLSTM->parameters.begin(),
                            bwdLSTM->parameters.end());
}


void BiLSTM::forward(const Eigen::MatrixXd & input) {
    fwdLSTM->forward(input);
    bwdLSTM->forward(input.rowwise().reverse());

    long hiddenDim = parameters[name+"_fwdLSTM_bi"]->rows();
    long sequenceLen = input.cols();
    Eigen::MatrixXd output(hiddenDim * 2, sequenceLen);

    output << fwdLSTM->cache[name+"_fwdLSTM_output"],
            bwdLSTM->cache[name+"_bwdLSTM_output"].rowwise().reverse();

    cache[name+"_output"] = output;
    cache[name+"_input"] = input;

    this->output = cache[name+"_output"];

}

void BiLSTM::backward(const Eigen::MatrixXd & dy) {
    cache[name+"_dy"] = dy;

    Eigen::MatrixXd fwdDy = dy.topRows(dy.rows()/2);
    fwdLSTM->backward(fwdDy);

    Eigen::MatrixXd bwdDy = dy.bottomRows(dy.rows()/2).rowwise().reverse();
    bwdLSTM->backward(bwdDy);

    diff.clear();
    diff.insert(fwdLSTM->diff.begin(), fwdLSTM->diff.end());
    diff.insert(bwdLSTM->diff.begin(), bwdLSTM->diff.end());

    diff[name+"_input"] = fwdLSTM->diff[name+"_fwdLSTM_input"] +
                          bwdLSTM->diff[name+"_bwdLSTM_input"].rowwise().reverse();

    this->inputDiff = diff[name+"_input"];
}


//
// Dropout implementation
//
Dropout::Dropout(float dropoutRate, std::string name) {
    this->dropoutRate = dropoutRate;
    this->name = name;
}

void Dropout::forward(const Eigen::MatrixXd & input) {
    Eigen::MatrixXd mask;
    if (dropoutRate >= 0) {
        // generate dropout mask
        int xSize = input.size();
        std::vector<double> v0(int(xSize * dropoutRate), 0);
        std::vector<double> v1(xSize - v0.size(), 1);
        v0.insert(v0.end(), v1.begin(), v1.end());
        std::random_shuffle(v0.begin(), v0.end());
        double *v_array = &v0[0];
        mask = Eigen::Map<Eigen::MatrixXd>(v_array, input.rows(), input.cols());
    } else {
        mask = cache[name+"_mask"];
    }

    // mask on x
    Eigen::MatrixXd y = input.array() * mask.array();

    cache[name+"_input"] = input;
    cache[name+"_output"] = y;
    cache[name+"_mask"] = mask;

    this->output = cache[name+"_output"];
}

void Dropout::backward(const Eigen::MatrixXd & dy) {
    cache[name+"_dy"] = dy;

    diff[name+"_input"] = dy.array() * cache[name+"_mask"].array();

    this->inputDiff = diff[name+"_input"];
}

void Dropout::gradientCheck() {
    auto input = cache[name+"_input"];
    auto dy = cache[name+"_dy"];

    std::cout << "=> " + name + " gradient checking..." << std::endl;

    int num_checks = 10;
    double delta = 10e-5;

    std::string paramName = name+"_input";
    auto& param = input;

    double originalDropoutRate = dropoutRate;
    dropoutRate = -1;
    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(param.rows()));
        int randCol = 0 + (rand() % (int)(param.cols()));

        double originalVal = param(randRow, randCol);

        param(randRow, randCol) = originalVal - delta;
        forward(input);
        auto output0 = cache[name+"_output"];

        param(randRow, randCol) = originalVal + delta;
        forward(input);
        auto output1 = cache[name+"_output"];

        param(randRow, randCol) = originalVal;

        double analyticGrad = diff[paramName](randRow, randCol);
        double numericalGrad =
                ((output1 - output0).array() * dy.array()).sum()
                / (2.0 * delta);
        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        if (rel_error > 10e-5)
            std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
    dropoutRate = originalDropoutRate;
}

//
// Crossentropy loss implementation
//
CrossEntropyLoss::CrossEntropyLoss (std::string name) {
    this->name = name;
}

void CrossEntropyLoss::forward(const Eigen::MatrixXd & pred,
                               const Eigen::MatrixXd & ref) {
    Eigen::MatrixXd loss = - ref.array() * pred.array().log();

    cache[name+"_pred"] = pred;
    cache[name+"_ref"] = ref;
    cache[name+"_output"] = loss;

    this->output = cache[name+"_output"];
}

void CrossEntropyLoss::backward() {
    Eigen::MatrixXd pred = cache[name+"_pred"];
    Eigen::MatrixXd ref = cache[name+"_ref"];

    Eigen::MatrixXd dpred = - ref.array() / pred.array();

    diff[name+"_input"] = dpred;

    this->inputDiff = diff[name+"_input"];
};

void CrossEntropyLoss::gradientCheck() {
    auto& pred = cache[name+"_pred"];
    auto& ref = cache[name+"_ref"];

    std::cout << "=> " + name + " gradient checking..." << std::endl;

    int num_checks = 10;
    double delta = 10e-5;

    std::string paramName = name+"_input";
    auto& param = pred;

    for (int i = 0; i < num_checks; ++i) {
        int randRow = 0 + (rand() % (int)(param.rows()));
        int randCol = 0 + (rand() % (int)(param.cols()));

        double originalVal = param(randRow, randCol);

        param(randRow, randCol) = originalVal - delta;
        forward(pred, ref);
        auto output0 = cache[name+"_output"];

        param(randRow, randCol) = originalVal + delta;
        forward(pred, ref);
        auto output1 = cache[name+"_output"];

        param(randRow, randCol) = originalVal;

        double analyticGrad = diff[paramName](randRow, randCol);
        double numericalGrad = (output1 - output0).sum() / (2.0 * delta);
        double rel_error = fabs(analyticGrad - numericalGrad) / fabs(analyticGrad + numericalGrad);

        if (rel_error > 10e-5)
            std::cout << "\t" << numericalGrad << ", " << analyticGrad << " ==> " << rel_error << std::endl;
    }
}