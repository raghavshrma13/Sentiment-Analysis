#pragma once
#include "preprocessor.hpp"
#include <string>

vector<string> preprocess_batch_serial(const vector<Tweet>& tweets, TextPreprocessor& processor);
void process_tweets_serial(const string& train_path, const string& test_path);