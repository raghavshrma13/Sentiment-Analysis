#pragma once
#include "preprocessor.hpp"
#include "parallel_encoder.hpp"
#include <chrono>

using namespace std;
using namespace std::chrono;

class SequentialProcessor {
public:
    SequentialProcessor();
    
    // Modified to return encodings along with timing
    struct ProcessResult {
        long preprocess_time;
        long encode_time;
        vector<vector<float>> encodings;
    };
    
    ProcessResult process_batch(const vector<Tweet>& tweets);
    void save_encodings(const string& filename, const vector<vector<float>>& encodings);

private:
    vector<string> preprocess_sequential(const vector<Tweet>& tweets);
    vector<vector<float>> encode_sequential(const vector<string>& texts);
    
    TextPreprocessor preprocessor;
    ParallelEncoder encoder;
};