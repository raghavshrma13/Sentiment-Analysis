#pragma once
#include "preprocessor.hpp"
#include <fstream>
#include <sstream>

vector<Tweet> load_tweets(const string& filename);
void save_processed_tweets(const string& filename, 
                         const vector<Tweet>& tweets,
                         const vector<string>& processed_texts);