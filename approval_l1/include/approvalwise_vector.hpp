#pragma once

#include <ios>
#include <utility>

#include "definitions.hpp"

std::pair<vector<approvalwise_vector_t>, size_t> load_approvalwise_vectors(
    std::istream& in);

int dist_l1(const approvalwise_vector_t& a, const approvalwise_vector_t& b);

int score_across(const vector<approvalwise_vector_t>& votings_hist,
                 const approvalwise_vector_t& voting, const int num_voters);

void print_vec(const vector<int>& v);