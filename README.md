# cDNN
DNN in C++.

### Compile:
Bi-LSTM tagger:
```
g++ -std=c++0x -O3 -I third_party/Eigen main.cpp bi_lstm.cpp loader.cpp nn.cpp utils.cpp -o bin/main
```

Bi-LSTM with character embedding tagger:
```
g++ -std=c++0x -O3 -I third_party/Eigen main.cpp bi_lstm_with_char.cpp.cpp loader.cpp nn.cpp utils.cpp -o bin/main
```

### Run
main <train_bio> <eval_bio> <pretrain_emb> <model_dir>

example:
```
main data/updated_UD_English/en-ud-train.bio data/updated_UD_English/en-ud-eval.bio /mnt/vol/gfsai-east/ai-group/users/trapit/wikidata/glove.6B.100d.txt model/
``` 

### Layer test:
Layer test randomly generates small training set and validate each layer with gradient check.
compile:
```
g++ -std=c++0x -O3 -I third_party/Eigen test.cpp loader.cpp nn.cpp utils.cpp -o bin/test
``` 
