#include "preprocessor.hpp"
#include <algorithm>

TextPreprocessor::TextPreprocessor(int num_threads) : num_threads_(num_threads) {
    if (num_threads_ <= 0) num_threads_ = 1;
    omp_set_num_threads(num_threads_);
}

vector<string> TextPreprocessor::preprocess_batch(const vector<Tweet>& tweets) {
    if (tweets.empty()) return vector<string>();
    
    vector<string> processed_texts(tweets.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tweets.size(); i++) {
        if (!tweets[i].text.empty()) {
            try {
                processed_texts[i] = clean_text(tweets[i].text);
            } catch (const exception& e) {
                #pragma omp critical
                {
                    cerr << "Error processing tweet " << tweets[i].id 
                         << ": " << e.what() << endl;
                    processed_texts[i] = tweets[i].text;
                }
            }
        } else {
            processed_texts[i] = "";
        }
    }
    
    return processed_texts;
}

string TextPreprocessor::clean_text(const string& text) {
    if (text.empty()) return "";
    
    string cleaned = text;
    
    // Remove URLs
    static const regex url_pattern(R"((https?:\/\/|www\.)[^\s]+)");
    cleaned = regex_replace(cleaned, url_pattern, "");
    
    // Remove mentions (@user)
    static const regex mention_pattern(R"(@\w+)");
    cleaned = regex_replace(cleaned, mention_pattern, "");
    
    to_lowercase(cleaned);
    remove_special_chars(cleaned);
    normalize_whitespace(cleaned);
    
    return cleaned;
}

void TextPreprocessor::to_lowercase(string& text) {
    transform(text.begin(), text.end(), text.begin(), ::tolower);
}

void TextPreprocessor::remove_special_chars(string& text) {
    static const regex special_chars(R"([^\w\s.,!?-])");
    text = regex_replace(text, special_chars, " ");
}

void TextPreprocessor::normalize_whitespace(string& text) {
    static const regex multi_spaces(R"(\s+)");
    text = regex_replace(text, multi_spaces, " ");
    
    if (!text.empty()) {
        size_t first = text.find_first_not_of(" \t\n\r");
        size_t last = text.find_last_not_of(" \t\n\r");
        if (first != string::npos && last != string::npos) {
            text = text.substr(first, last - first + 1);
        } else {
            text = "";
        }
    }
}