# cDNN
DNN in C++.

Bi-LSTM tagger:
compile: g++ -std=c++0x -O3 -I lib/eigen-3.3.3/Eigen main.cpp bi_lstm.cpp loader.cpp nn.cpp utils.cpp -o bin/main

Bi-LSTM with character embedding tagger:
compile: g++ -std=c++0x -O3 -I lib/eigen-3.3.3/Eigen main.cpp bi_lstm_with_char.cpp.cpp loader.cpp nn.cpp utils.cpp -o bin/main

run: main <train_bio> <eval_bio> <pretrain_emb> <model_dir>
example: main data/updated_UD_English/en-ud-train.bio data/updated_UD_English/en-ud-eval.bio /mnt/vol/gfsai-east/ai-group/users/trapit/wikidata/glove.6B.100d.txt model/

Layer test:
compile: g++ -std=c++0x -O3 -I lib/eigen-3.3.3/Eigen test.cpp loader.cpp nn.cpp utils.cpp -o bin/test
