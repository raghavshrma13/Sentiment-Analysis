#include "preprocessor.hpp"
#include "sequential.hpp"
#include "common_headers.hpp"
#include "parallel_encoder.hpp"  // Add this line
#include <chrono>

using namespace std::chrono;

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
    
    // Read tweets
    while (getline(file, line)) {
        line_num++;
        try {
            vector<string> fields;
            string current_field;
            bool in_quotes = false;
            
            for (size_t i = 0; i < line.length(); i++) {
                char c = line[i];
                if (c == '"') {
                    in_quotes = !in_quotes;
                }
                else if (c == ',' && !in_quotes) {
                    // Trim whitespace from field
                    while (!current_field.empty() && isspace(current_field.back())) {
                        current_field.pop_back();
                    }
                    while (!current_field.empty() && isspace(current_field.front())) {
                        current_field.erase(0, 1);
                    }
                    fields.push_back(current_field);
                    current_field.clear();
                }
                else {
                    current_field += c;
                }
            }
            // Add last field
            fields.push_back(current_field);

            // Create Tweet object
            if (fields.size() >= 3) {
                Tweet tweet;
                tweet.id = line_num;
                
                // Get text (remove surrounding quotes if present)
                string text = fields[1];
                if (text.length() >= 2 && text.front() == '"' && text.back() == '"') {
                    text = text.substr(1, text.length() - 2);
                }
                tweet.text = text;
                
                // Get sentiment (last field, remove quotes)
                string sentiment_str = fields.back();
                if (sentiment_str.length() >= 2 && 
                    sentiment_str.front() == '"' && 
                    sentiment_str.back() == '"') {
                    sentiment_str = sentiment_str.substr(1, sentiment_str.length() - 2);
                }
                
                // Clean up sentiment string
                sentiment_str.erase(
                    remove_if(sentiment_str.begin(), sentiment_str.end(), 
                            [](char c) { return isspace(c); }),
                    sentiment_str.end()
                );
                
                // Map sentiment values, default to neutral (1) silently
                if (sentiment_str == "Positive" || sentiment_str == "2") {
                    tweet.sentiment = 2;
                } else if (sentiment_str == "Negative" || sentiment_str == "0") {
                    tweet.sentiment = 0;
                } else if (sentiment_str == "Irrelevant" || sentiment_str == "-1") {
                    tweet.sentiment = -1;
                } else {
                    tweet.sentiment = 1;  // Default to neutral without warning
                }
                tweets.push_back(tweet);
            }
        }
        catch (const exception& e) {
            cerr << "Warning: Skipping malformed line " << line_num << endl;
            continue;
        }
    }
    
    cout << "Successfully loaded " << tweets.size() 
         << " tweets from " << filename << endl;
    
    return tweets;
}

void save_processed_tweets(const string& filename, 
                         const vector<Tweet>& tweets,
                         const vector<string>& processed_texts) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open output file " << filename << endl;
        return;
    }
    
    file << "id,text,sentiment\n";
    
    for (size_t i = 0; i < tweets.size(); i++) {
        if (i < processed_texts.size()) {  // Safety check
            file << tweets[i].id << ","
                 << "\"" << processed_texts[i] << "\","
                 << tweets[i].sentiment << "\n";
        }
    }
    file.close();
}

void save_embeddings(const string& filename, const vector<vector<float>>& embeddings) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open output file " << filename << endl;
        return;
    }
    
    // Save number of embeddings and embedding size
    size_t num_embeddings = embeddings.size();
    size_t embedding_size = embeddings.empty() ? 0 : embeddings[0].size();
    
    file.write(reinterpret_cast<const char*>(&num_embeddings), sizeof(num_embeddings));
    file.write(reinterpret_cast<const char*>(&embedding_size), sizeof(embedding_size));
    
    // Save embeddings
    for (const auto& embedding : embeddings) {
        file.write(reinterpret_cast<const char*>(embedding.data()), 
                  embedding.size() * sizeof(float));
    }
    
    cout << "Saved " << num_embeddings << " embeddings of size " 
         << embedding_size << " to " << filename << endl;
}

int main() {

    
    
    
    try {
        cout << "RAGHAV SHARMA 2023BCS0050 GAURAV JHALANI 2023BCS0032" << endl;
        const string train_path = "data/raw/train_for_cpp.csv";
        const string test_path = "data/raw/test_for_cpp.csv";

        // sizes to test (number of tweets to process)
        vector<size_t> sizes = {100, 1000, 10000};
        const int num_threads = 8;

        // load full training set once
        cout << "Processing training data..." << endl;
        auto train_tweets = load_tweets(train_path);
        if (train_tweets.empty()) {
            cerr << "No training tweets loaded" << endl;
            return 1;
        }

        for (size_t n : sizes) {
            size_t use_n = min(n, train_tweets.size());
            cout << "\n=== Testing dataset size: " << use_n << " ===" << endl;

            // prepare subset of tweets
            vector<Tweet> subset_tweets(train_tweets.begin(), train_tweets.begin() + use_n);

            // 1) Parallel preprocessing timing
            TextPreprocessor preprocessor(num_threads);
            auto t_pre_start = high_resolution_clock::now();
            vector<string> processed_texts = preprocessor.preprocess_batch(subset_tweets);
            auto t_pre_end = high_resolution_clock::now();
            auto pre_ms = duration_cast<milliseconds>(t_pre_end - t_pre_start).count();

            cout << "Parallel preprocessing time: " << pre_ms << " ms" << endl;

            // 2) Parallel one-hot vocabulary build + embedding timing
            ParallelEncoder encoder(5000, num_threads);
            encoder.build_vocabulary(processed_texts); // build vocab from preprocessed texts

            auto t_emb_start = high_resolution_clock::now();
            auto encodings = encoder.encode_parallel(processed_texts);
            auto t_emb_end = high_resolution_clock::now();
            auto emb_ms = duration_cast<milliseconds>(t_emb_end - t_emb_start).count();

            cout << "Parallel embedding time: " << emb_ms << " ms" << endl;
            cout << "Vocabulary size used: " << encoder.get_vocab_size() << endl;
            cout << "Encoded vectors: " << encodings.size() << " x " 
                 << (encodings.empty() ? 0 : encodings[0].size()) << endl;
        }

    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }

    return 0;
}