BIN_DIR=../bin

CC = g++
CFLAGS = -I ../third_party/eigen -pthread -O3 -std=c++11

all: seqLabelingTrainer seqLabelingTagger

seqLabelingTrainer : trainer.cpp
	$(CC) trainer.cpp loader.cpp charBiLSTMNet.cpp ../net.cpp ../nn.cpp ../utils.cpp -o ${BIN_DIR}/seqLabelingTrainer $(CFLAGS)
seqLabelingTagger : tagger.cpp
	$(CC) tagger.cpp loader.cpp charBiLSTMNet.cpp ../net.cpp ../nn.cpp ../utils.cpp -o ${BIN_DIR}/seqLabelingTagger $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf seqLabelingTrainer seqLabelingTagger; popd