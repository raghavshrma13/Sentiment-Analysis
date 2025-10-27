#include "sequential_processor.hpp"
#include <iostream>
#include <stdexcept>

SequentialProcessor::SequentialProcessor() 
    : preprocessor(1), encoder(5000, 1) {} // Single thread

vector<string> SequentialProcessor::preprocess_sequential(const vector<Tweet>& tweets) {
    vector<string> processed_texts;
    processed_texts.reserve(tweets.size());
    
    for (const auto& tweet : tweets) {
        processed_texts.push_back(preprocessor.clean_text(tweet.text));
    }
    return processed_texts;
}

vector<vector<float>> SequentialProcessor::encode_sequential(const vector<string>& texts) {
    vector<vector<float>> encodings;
    encodings.reserve(texts.size());
     
    // Build vocabulary first
    encoder.build_vocabulary(texts);
    
    // Sequential encoding
    return encoder.encode_sequential(texts);
}

SequentialProcessor::ProcessResult SequentialProcessor::process_batch(const vector<Tweet>& tweets) {
    ProcessResult result;

    // Time preprocessing
    auto pre_start = high_resolution_clock::now();
    auto processed_texts = preprocess_sequential(tweets);
    auto pre_end = high_resolution_clock::now();
    result.preprocess_time = duration_cast<milliseconds>(pre_end - pre_start).count();

    // Time encoding
    auto enc_start = high_resolution_clock::now();
    result.encodings = encode_sequential(processed_texts);
    auto enc_end = high_resolution_clock::now();
    result.encode_time = duration_cast<milliseconds>(enc_end - enc_start).count();

    // Optional summary prints
    cout << "Processed texts: " << processed_texts.size() << endl;
    cout << "Encoded vectors: " << result.encodings.size() << " x "
         << (result.encodings.empty() ? 0 : result.encodings[0].size()) << endl;

    return result;
}

void SequentialProcessor::save_encodings(const string& filename, const vector<vector<float>>& encodings) {
    // forward to internal encoder's saver
    try {
        encoder.save_encodings(filename, encodings);
    } catch (const std::exception& e) {
        std::cerr << "Error saving encodings to " << filename << ": " << e.what() << std::endl;
    }
}