#include "parallel_encoder.hpp"
#include <algorithm>
#include <regex>
#include <iostream>
#include <chrono>

using namespace std::chrono;

ParallelEncoder::ParallelEncoder(int vocab_size, int threads) 
    : max_vocab_size(vocab_size), num_threads(threads) {
    omp_set_num_threads(num_threads);
}

vector<string> ParallelEncoder::tokenize(const string& text) const {
    string cleaned = text;
    transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
    
    regex pattern("[^a-z0-9 ]");
    cleaned = regex_replace(cleaned, pattern, " ");
    
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = cleaned.find(" ", start)) != string::npos) {
        if (end > start) {
            tokens.push_back(cleaned.substr(start, end - start));
        }
        start = end + 1;
    }
    if (start < cleaned.length()) {
        tokens.push_back(cleaned.substr(start));
    }
    return tokens;
}

void ParallelEncoder::build_vocabulary(const vector<string>& texts) {
    unordered_map<string, int> word_freq;
    
    #pragma omp parallel
    {
        unordered_map<string, int> local_freq;
        
        #pragma omp for schedule(dynamic)
        for (const auto& text : texts) {
            auto tokens = tokenize(text);
            for (const auto& token : tokens) {
                local_freq[token]++;
            }
        }
        
        #pragma omp critical
        {
            for (const auto& pair : local_freq) {
                word_freq[pair.first] += pair.second;
            }
        }
    }
    
    vector<pair<string, int>> word_list(word_freq.begin(), word_freq.end());
    sort(word_list.begin(), word_list.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    vocabulary.clear();
    for (size_t i = 0; i < min(static_cast<size_t>(max_vocab_size), word_list.size()); i++) {
        vocabulary[word_list[i].first] = i;
    }
}

vector<vector<float>> ParallelEncoder::encode_parallel(const vector<string>& texts) {
    auto start_time = high_resolution_clock::now();
    
    vector<vector<float>> encodings(texts.size(), vector<float>(vocabulary.size(), 0.0f));
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < texts.size(); i++) {
        auto tokens = tokenize(texts[i]);
        for (const auto& token : tokens) {
            auto it = vocabulary.find(token);
            if (it != vocabulary.end()) {
                encodings[i][it->second] = 1.0f;
            }
        }
        
        #pragma omp critical
        {
            if (i % 1000 == 0) {
                cout << "Processed " << i << " texts..." << endl;
            }
        }
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Parallel encoding took: " << duration.count() << " milliseconds" << endl;
    
    return encodings;
}

vector<vector<float>> ParallelEncoder::encode_sequential(const vector<string>& texts) {
    auto start_time = high_resolution_clock::now();
    
    vector<vector<float>> encodings(texts.size(), vector<float>(vocabulary.size(), 0.0f));
    
    for (size_t i = 0; i < texts.size(); i++) {
        auto tokens = tokenize(texts[i]);
        for (const auto& token : tokens) {
            auto it = vocabulary.find(token);
            if (it != vocabulary.end()) {
                encodings[i][it->second] = 1.0f;
            }
        }
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Sequential encoding took: " << duration.count() << " milliseconds" << endl;
    
    return encodings;
}