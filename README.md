# cDNN
A bi-directional LSTM name tagger with character embeddings written in C++.

### Compile
trainer:
```
g++ -std=c++11 -O3 -pthread -I third_party/eigen/ trainer.cpp loader.cpp nn.cpp utils.cpp charBiLSTMNet.cpp net.cpp -o bin/trainer
```

tagger:
```
g++ -std=c++11 -O3 -pthread -I third_party/eigen/ tagger.cpp loader.cpp nn.cpp utils.cpp charBiLSTMNet.cpp net.cpp -o bin/tagger
```

### Train
trainer <train_bio> <eval_bio> <pretrain_emb> <model_dir>

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
