#pragma once
#include <omp.h>
#include <string>
#include <vector>
#include <iostream>
#include <regex>

using namespace std;

struct Tweet {
    int id;
    string text;
    int sentiment;
};

class TextPreprocessor {
public:
    TextPreprocessor(int num_threads = 4);
    vector<string> preprocess_batch(const vector<Tweet>& tweets);
    string clean_text(const string& text);

private:
    
    void to_lowercase(string& text);
    void remove_special_chars(string& text);
    void normalize_whitespace(string& text);
    int num_threads_;
};