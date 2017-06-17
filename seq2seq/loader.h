//
// Created by Boliang Zhang on 6/10/17.
//

#ifndef CDNN_LOADER_H
#define CDNN_LOADER_H

#include "../utils.h"
#include "seq2seq.h"
#include <vector>
#include <Eigen/Core>
#include <map>

void loadRawData(std::string & filePath,
                 std::vector<InputSeq2Seq> & data,
                 bool isTrain);

void processData(InputSeq2Seq & s,
                 const Eigen::MatrixXd& srcEmbedding,
                 const Eigen::MatrixXd& trgEmbedding);

void createDate(std::vector<InputSeq2Seq> & data,
                const std::map<std::string, int>& trgToken2Id);

void loadMapping(std::string & modelDir,
                 std::map<std::string, int> & srcToken2Id,
                 std::map<int, std::string> & srcId2Token,
                 std::map<std::string, int> & trgToken2Id,
                 std::map<int, std::string> & trgId2Token);

#endif //CDNN_LOADER_H
