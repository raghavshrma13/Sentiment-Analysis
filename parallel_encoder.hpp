#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <fstream>

using namespace std;

class ParallelEncoder {
public:
    ParallelEncoder(int vocab_size = 5000, int num_threads = 8);
    void build_vocabulary(const vector<string>& texts);
    vector<vector<float>> encode_parallel(const vector<string>& texts);
    vector<vector<float>> encode_sequential(const vector<string>& texts);
    int get_vocab_size() const { return vocabulary.size(); }
    void save_encodings(const string& filename, const vector<vector<float>>& encodings) const {
        ofstream file(filename, ios::binary);
        if (!file) {
            throw runtime_error("Cannot open file for writing: " + filename);
        }
        
        size_t n = encodings.size();
        size_t dim = encodings.empty() ? 0 : encodings[0].size();
        
        file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        
        for (const auto& encoding : encodings) {
            file.write(reinterpret_cast<const char*>(encoding.data()), 
                      encoding.size() * sizeof(float));
        }
    }

private:
    vector<string> tokenize(const string& text) const;
    unordered_map<string, int> vocabulary;
    int max_vocab_size;
    int num_threads;
};