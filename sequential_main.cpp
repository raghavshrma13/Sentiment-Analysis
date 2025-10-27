#include "sequential_processor.hpp"
#include "preprocessor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std;
using namespace std::chrono;

// Implementation of load_tweets
vector<Tweet> load_tweets(const string& filename) {
    vector<Tweet> tweets;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return tweets;
    }
    
    string line;
    int line_num = 0;
    
    // Skip header
    getline(file, line);
    
    while (getline(file, line)) {
        line_num++;
        stringstream ss(line);
        vector<string> fields;
        string field;
        
        while (getline(ss, field, ',')) {
            fields.push_back(field);
        }
        
        if (fields.size() >= 3) {
            Tweet tweet;
            tweet.id = line_num;
            
            string text = fields[1];
            if (text.length() >= 2 && text.front() == '"' && text.back() == '"') {
                text = text.substr(1, text.length() - 2);
            }
            tweet.text = text;
            
            string sentiment_str = fields.back();
            if (sentiment_str.length() >= 2 && 
                sentiment_str.front() == '"' && 
                sentiment_str.back() == '"') {
                sentiment_str = sentiment_str.substr(1, sentiment_str.length() - 2);
            }
            
            sentiment_str.erase(
                remove_if(sentiment_str.begin(), sentiment_str.end(), 
                        [](char c) { return isspace(c); }),
                sentiment_str.end()
            );
            
            if (sentiment_str == "Positive" || sentiment_str == "2") {
                tweet.sentiment = 2;
            } else if (sentiment_str == "Negative" || sentiment_str == "0") {
                tweet.sentiment = 0;
            } else if (sentiment_str == "Irrelevant" || sentiment_str == "-1") {
                tweet.sentiment = -1;
            } else {
                tweet.sentiment = 1;  // Default to neutral
            }
            tweets.push_back(tweet);
        }
    }
    
    cout << "Successfully loaded " << tweets.size() << " tweets from " << filename << endl;
    return tweets;
}

int main() {

    cout<< "RAGHAV SHARMA 2023BCS0050 GAURAV JHALANI 2023BCS0032" << endl;
    try {
        const string train_path = "data/raw/train_for_cpp.csv";
        vector<size_t> sizes = {100, 1000, 10000};
        
        // Load full dataset
        cout << "Loading training data..." << endl;
        auto train_tweets = load_tweets(train_path);
        if (train_tweets.empty()) {
            cerr << "No training tweets loaded" << endl;
            return 1;
        }
        
        SequentialProcessor processor;
        
        // Process each batch size
        for (size_t n : sizes) {
            size_t use_n = min(n, train_tweets.size());
            cout << "\n=== Testing dataset size: " << use_n << " ===" << endl;
            
            // Prepare subset of tweets
            vector<Tweet> subset_tweets(train_tweets.begin(), train_tweets.begin() + use_n);
            
            // Process and generate encodings
            auto result = processor.process_batch(subset_tweets);
            
            cout << "Sequential preprocessing time: " << result.preprocess_time << " ms" << endl;
            cout << "Sequential embedding time: " << result.encode_time << " ms" << endl;
            
            // Save the encodings
            string out_path = "data/embeddings/train_onehot_" + to_string(use_n) + ".bin";
            processor.save_encodings(out_path, result.encodings);
            
            cout << "Saved encodings to: " << out_path << endl;
        }
        
    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}