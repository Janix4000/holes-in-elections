#include "approvalwise_vector.hpp"

std::pair<vector<approvalwise_vector_t>, size_t> load_approvalwise_vectors(
    std::istream& in) {
    size_t num_vectors, num_voters, num_candidates;
    in >> num_vectors >> num_voters >> num_candidates;
    vector<approvalwise_vector_t> res(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        std::string instance_id;
        in >> instance_id;
        approvalwise_vector_t v(num_candidates);
        for (size_t j = 0; j < num_candidates; ++j) {
            in >> v[j];
        }
        res[i] = v;
    }

    return {
        std::move(res),
        num_voters,
    };
}

void save_approvalwise_vectors(std::ostream& out,
                               const vector<approvalwise_vector_t>& vectors,
                               const size_t num_voters) {
    size_t num_vectors = vectors.size();
    size_t num_candidates = vectors[0].size();

    out << num_vectors << " " << num_voters << " " << num_candidates << "\n";
    size_t idx = 0;
    for (const auto& v : vectors) {
        out << idx++ << " ";
        for (const auto& val : v) {
            out << val << " ";
        }
        out << "\n";
    }
}

int dist_l1(const approvalwise_vector_t& a, const approvalwise_vector_t& b) {
    int dist = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dist += abs(a[i] - b[i]);
    }
    return dist;
}

int score_across(const vector<approvalwise_vector_t>& votings_hist,
                 const approvalwise_vector_t& voting, const int num_voters) {
    const int num_candidates = votings_hist.front().size();
    int min_score = num_candidates * num_voters + 1;
    for (const auto& voting_hist : votings_hist) {
        min_score = min(min_score, dist_l1(voting, voting_hist));
    }
    return min_score;
}

void print_vec(const vector<int>& v) {
    cout << "[";
    if (v.size()) {
        cout << v.front();
    }
    for (size_t i = 1; i < v.size(); i++) {
        cout << ", " << v[i];
    }
    cout << "]";
}