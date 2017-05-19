cmake_minimum_required(VERSION 3.7)
project(cDNN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(SOURCE_FILES main.cpp lstm.h utils.h mlp.h dropout.h loss.h bi_lstm_with_char.h loader.h)
include_directories( lib/eigen-3.3.3/Eigen )

add_executable(cDNN ${SOURCE_FILES})