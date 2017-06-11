//
// Created by Boliang Zhang on 6/10/17.
//

SeqToSeq::SeqToSeq(const std::map<std::string, std::string> & configuration);
SeqToSeq::SeqToSeq(const std::map<std::string, std::string> & configuration,
         const std::map<std::string, Eigen::MatrixXd*>& parameters);

void SeqToSeq::forward(const Sequence & input);
void SeqToSeq::forward(const Sequence & input, bool isTrain);

void SeqToSeq::backward();

void SeqToSeq::update();