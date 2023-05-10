#pragma once

#include "definitions.hpp"

int dist_l1(const voting_hist_t& a, const voting_hist_t& b) {
    int dist = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dist += abs(a[i] - b[i]);
    }
    return dist;
}

int score_across(const vector<voting_hist_t>& votings_hist,
                 const voting_hist_t& voting, const int N) {
    const int M = votings_hist.front().size();
    int min_score = M * N + 1;
    for (const auto& voting_hist : votings_hist) {
        min_score = min(min_score, dist_l1(voting, voting_hist));
    }
    return min_score;
}