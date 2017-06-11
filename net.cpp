//
// Created by Boliang Zhang on 5/30/17.
//
#include "net.h"
#include "utils.h"


void Net::gradientCheck(Sequence & input){
    std::cout << "=> Network gradient checking..." << std::endl;

    int num_checks = 10;
    double delta = 10e-5;

    std::vector<std::pair<std::string, Eigen::MatrixXd*> > paramToCheck;
    std::map<std::string, Eigen::MatrixXd*>::iterator it;
    for (it = parameters.begin(); it != parameters.end(); ++it)
        paramToCheck.push_back({it->first, it->second});
    paramToCheck.push_back({name+"_wordInput", & input.wordEmb});

    std::cout.precision(15);
    for (int i = 0; i < paramToCheck.size(); ++i) {
        auto paramName = paramToCheck[i].first;
        auto param = paramToCheck[i].second;
        auto paramDiff = diff[paramName];

        std::printf("checking %s %s\n", name.c_str(), paramName.c_str());

        assert(param->rows() == diff[paramName].rows()
               || param->cols() == diff[paramName].cols());

        for (int j = 0; j < num_checks; ++j) {
            int randRow = rand() % (int)(param->rows());
            int randCol = rand() % (int)(param->cols());

            double originalVal = (*param)(randRow, randCol);

            (*param)(randRow, randCol) = originalVal - delta;
            forward(input, false);
            auto output0 = cache[name+"_output"];

            (*param)(randRow, randCol) = originalVal + delta;
            forward(input, false);
            auto output1 = cache[name+"_output"];

            (*param)(randRow, randCol) = originalVal;

            double analyticGrad = diff[paramName](randRow, randCol);
            double numericalGrad;
            numericalGrad = (output1 - output0).array().sum() / (2.0 * delta);
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

void Net::updateEmbedding(Eigen::MatrixXd* embDict,
                          const std::vector<int> & wordIndex){
    mtx.lock();
    float learningRate = std::stof(configuration["learningRate"]);
    gradientClip(diff[name+"_wordInput"]);
    for (int i = 0; i < wordIndex.size(); ++i) {
        Eigen::MatrixXd dWord = diff[name+"_wordInput"].col(i);
        embDict->col(wordIndex[i]) -= learningRate * dWord;
    }
    mtx.unlock();
}

void Net::loadNet(std::string modelDir,
                  std::map<std::string, std::string>& configuration,
                  std::map<std::string, Eigen::MatrixXd*>& parameters,
                  std::map<std::string, int>& word2id,
                  std::map<std::string, int>& char2id,
                  std::map<std::string, int>& label2id,
                  std::map<int, std::string>& id2word,
                  std::map<int, std::string>& id2char,
                  std::map<int, std::string>& id2label,
                  Eigen::MatrixXd& wordEmbedding,
                  Eigen::MatrixXd& charEmbedding){

    std::string line;
    std::vector<std::string> lineValues;

    // load configuration from file
    std::cout << "loading net configuration...";
    std::ifstream ifsConfig(modelDir + "/configuration.mdl");
    while (std::getline(ifsConfig, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, '=')) {
            lineValues.push_back(cell);
        }
        configuration[lineValues[0]] = lineValues[1];
        lineValues.clear();
    }
    std::cout << "Done" << std::endl;

    // load parameters from file
    std::cout << "loading net parameters...";

    std::ifstream ifsParameters(modelDir + "/parameters.mdl");
    while (std::getline(ifsParameters, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, '|')) {
            lineValues.push_back(cell);
        }
        auto& paramName = lineValues[0];
        int rows = std::stoi(lineValues[1]);
        int cols = std::stoi(lineValues[2]);
        auto& data = lineValues[3];

        std::stringstream dataStream(data);
        double dataValues[rows*cols];
        int i = 0;
        while (std::getline(dataStream, cell, ' ')) {
            dataValues[i] = std::stof(cell);
            i++;
        }

        parameters[paramName] = new Eigen::MatrixXd(rows, cols);;
        *parameters[paramName] = Eigen::Map<Eigen::MatrixXd>(dataValues, rows, cols);

        lineValues.clear();
    }
    std::cout << "Done" << std::endl;

    // load mapping from file
    std::cout << "loading net mapping...";

    std::ifstream ifsWord2Id(modelDir + "/word2id.mdl");
    while (std::getline(ifsWord2Id, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        word2id[lineValues[0]] = std::stoi(lineValues[1]);
        lineValues.clear();
    }

    std::ifstream ifsChar2Id(modelDir + "/char2id.mdl");
    while (std::getline(ifsChar2Id, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        char2id[lineValues[0]] = std::stoi(lineValues[1]);
        lineValues.clear();
    }

    std::ifstream ifsLabel2Id(modelDir + "/label2id.mdl");
    while (std::getline(ifsLabel2Id, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        label2id[lineValues[0]] = std::stoi(lineValues[1]);
        lineValues.clear();
    }

    std::ifstream ifsId2Word(modelDir + "/id2word.mdl");
    while (std::getline(ifsId2Word, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        id2word[std::stoi(lineValues[0])] = lineValues[1];
        lineValues.clear();
    }

    std::ifstream ifsId2Char(modelDir + "/id2char.mdl");
    while (std::getline(ifsId2Char, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        id2char[std::stoi(lineValues[0])] = lineValues[1];
        lineValues.clear();
    }

    std::ifstream ifsId2Label(modelDir + "/id2label.mdl");
    while (std::getline(ifsId2Label, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ' ')) {
            lineValues.push_back(cell);
        }
        id2label[std::stoi(lineValues[0])] = lineValues[1];
        lineValues.clear();
    }
    std::cout << "Done" << std::endl;

    // load word embeddings from file
    std::cout << "loading word embeddings...";
    std::string wordEmbedingFile = modelDir + "/wordEmbedding.mdl";
    wordEmbedding = readMatrix(wordEmbedingFile.c_str());
    std::cout << "Done" << std::endl;

    // load char embeddings from file
    std::cout << "loading char embeddings...";
    std::string charEmbedingFile = modelDir + "/charEmbedding.mdl";
    charEmbedding = readMatrix(charEmbedingFile.c_str());
    std::cout << "Done" << std::endl;
}


void Net::saveNet(const std::map<std::string, std::string>& configuration,
                  const std::map<std::string, Eigen::MatrixXd*> parameters,
                  const std::map<std::string, int>& word2id,
                  const std::map<std::string, int>& char2id,
                  const std::map<std::string, int>& label2id,
                  const std::map<int, std::string>& id2word,
                  const std::map<int, std::string>& id2char,
                  const std::map<int, std::string>& id2label,
                  const Eigen::MatrixXd& wordEmbedding,
                  const Eigen::MatrixXd& charEmbedding){
    auto& modelDir = configuration.at("modelDir");

    // write configuration to file
    std::ofstream ofsConfig(modelDir + "/configuration.mdl");
    for (const auto & p: configuration)
        ofsConfig << p.first << "=" << p.second << std::endl;
    ofsConfig.close();

    // write parameters to file
    std::ofstream ofsParameters(modelDir + "/parameters.mdl");
    for (const auto & p: parameters){
        auto & param = *(p.second);
        ofsParameters << p.first
                      << "|"
                      << param.rows()
                      << "|"
                      << param.cols()
                      << "|";
        for (int i = 0; i < param.rows() * param.cols(); ++i)
            ofsParameters << param.data()[i] << " ";
        ofsParameters << std::endl;
    }
    ofsParameters.close();

    // write mapping to file
    std::ofstream ofsWord2Id(modelDir + "/word2id.mdl");
    for (const auto & p: word2id){
        ofsWord2Id << p.first << " " << p.second << std::endl;
    }
    ofsWord2Id.close();

    std::ofstream ofsChar2Id(modelDir + "/char2id.mdl");
    for (const auto & p: char2id){
        ofsChar2Id << p.first << " " << p.second << std::endl;
    }
    ofsChar2Id.close();

    std::ofstream ofsLabel2Id(modelDir + "/label2id.mdl");
    for (const auto & p: label2id){
        ofsLabel2Id << p.first << " " << p.second << std::endl;
    }
    ofsLabel2Id.close();

    std::ofstream ofsId2Word(modelDir + "/id2word.mdl");
    for (const auto & p: id2word){
        ofsId2Word << p.first << " " << p.second << std::endl;
    }
    ofsId2Word.close();

    std::ofstream ofsId2Char(modelDir + "/id2char.mdl");
    for (const auto & p: id2char){
        ofsId2Char << p.first << " " << p.second << std::endl;
    }
    ofsId2Char.close();

    std::ofstream ofsId2Label(modelDir + "/id2label.mdl");
    for (const auto & p: id2label){
        ofsId2Label << p.first << " " << p.second << std::endl;
    }
    ofsId2Label.close();

    // write word embeddings to file
    std::ofstream ofsWordEmb(modelDir + "/wordEmbedding.mdl");
    ofsWordEmb << wordEmbedding << std::endl;
    ofsWordEmb.close();

    // write char embeddings to file
    std::ofstream ofsCharEmb(modelDir + "/charEmbedding.mdl");
    ofsCharEmb << charEmbedding << std::endl;
    ofsCharEmb.close();
}

Eigen::MatrixXd readMatrix(const char *filename)
{
    int cols = 0, rows = 0;
//    static double buff[MAXBUFSIZE];
    std::vector<double> buff;

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(filename);
    while (! infile.eof())
    {
        std::string line;
        std::getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
        while(! stream.eof()){
            double a;
            stream >> a;
            buff.push_back(a);
            temp_cols++;
        }


        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    buff.clear();

    return result;
};