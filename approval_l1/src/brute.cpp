#include "brute.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

signed main(int argc, char** args) {
    if (argc != 3 + 1) {
        cerr << "Usage: " << args[0]
             << " N(voters) M(candidates) R(additional votings)" << endl;
        return 1;
    }
    const int N = stoi(args[1]);
    const int M = stoi(args[2]);
    const int R = stoi(args[3]);
    vector<voting_hist_t> votings_hist = {vi(M, N), vi(M, 0)};
    const size_t start_N = votings_hist.size();

    cout << "r,dist,dist_prop,time" << endl;
    for (int i = 0; i < R; i++) {
        auto start_time = chrono::high_resolution_clock::now();
        auto [res, score] = brute::next_voting_hist(votings_hist, N);
        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time =
            chrono::duration_cast<chrono::milliseconds>(end_time - start_time)
                .count();
        cout << i + start_N + 1 << ',' << score << ','
             << double(score) / (N * M) << ',' << setprecision(4)
             << double(elapsed_time / 1000.) << endl;

        print_vec(res);
        cout << endl;

        votings_hist.push_back(res);
    }
}