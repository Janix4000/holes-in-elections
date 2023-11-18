#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "approvalwise_vector.hpp"

using namespace std;

namespace greedy_dp {

pair<approvalwise_vector_t, int> farthest_approvalwise_vector(
    const vector<approvalwise_vector_t>& approvalwise_vectors,
    const int voters_num) {
    const int elections_num = approvalwise_vectors.size();
    const int candidates_num = approvalwise_vectors.front().size();
    const int INF = voters_num * candidates_num + 1;

    vvvi dp_per_voting(candidates_num,
                       vvi(voters_num + 1, vi(elections_num, 0)));
    vvi dp(candidates_num, vi(voters_num + 1, 0));
    vvi from(candidates_num, vi(voters_num + 1, -1));

    for (int y = voters_num; y >= 0; --y) {
        const int x = 0;
        for (int r = 0; r < elections_num; r++) {
            dp_per_voting[x][y][r] = abs(y - approvalwise_vectors[r][x]);
        }
        dp[x][y] = *min_element(all(dp_per_voting[x][y]));
    }

    for (int x = 1; x < candidates_num; x++) {
        for (int y = voters_num; y >= 0; --y) {
            int max_y = -1;
            int max_for_y = -1;
            const int x_prev = x - 1;
            for (int y_prev = voters_num; y_prev >= y; y_prev--) {
                int min_for_r = INF;
                for (int r = 0; r < elections_num; r++) {
                    const int val = abs(y - approvalwise_vectors[r][x]) +
                                    dp_per_voting[x_prev][y_prev][r];
                    min_for_r = min(min_for_r, val);
                }
                if (min_for_r > max_for_y) {
                    max_for_y = min_for_r;
                    max_y = y_prev;
                }
            }

            for (int r = 0; r < elections_num; r++) {
                dp_per_voting[x][y][r] = abs(y - approvalwise_vectors[r][x]) +
                                         dp_per_voting[x_prev][max_y][r];
            }

            dp[x][y] = *min_element(all(dp_per_voting[x][y]));
            from[x][y] = max_y;
        }
    }

    approvalwise_vector_t res(candidates_num);
    auto max_el = max_element(all(dp.back()));
    int y = max_el - dp.back().begin();
    res.back() = y;
    for (int x = candidates_num - 2; x >= 0; x--) {
        y = from[x + 1][y];
        res[x] = y;
    }

    // for (int y = N; y >= 0; y--) {
    //   cout << setw(2) << setfill('0') << y << ": ";
    //   cout << "[";
    //   cout << setw(3) << setfill('0') << dp[0][y];
    //   for (size_t i = 1; i < M; i++) {
    //     cout << ", " << setw(3) << setfill('0') << dp[i][y];
    //   }
    //   cout << "]\n";
    // }
    // for (int y = N; y >= 0; y--) {
    //   cout << setw(2) << setfill('0') << y << ": ";
    //   cout << "[";
    //   cout << setw(2) << setfill('0') << from[0][y];
    //   for (size_t i = 1; i < M; i++) {
    //     cout << ", " << setw(2) << setfill('0') << from[i][y];
    //   }
    //   cout << "]\n";
    // }

    return {res, *max_el};
}
}  // namespace greedy_dp

// inline signed main(int argc, char** args) {
//     if (argc != 3 + 1) {
//         cerr << "Usage: " << args[0]
//              << " N(voters) M(candidates) R(elections to generate)" << endl;
//         return 1;
//     }
//     const int num_voters = stoi(args[1]);
//     const int num_candidates = stoi(args[2]);
//     const int num_elections_to_generate = stoi(args[3]);
//     vector<approvalwise_vector_t> votings_hist = {
//         vi(num_candidates, num_voters), vi(num_candidates, 0)};

//     cout << "r,dist,dist_prop,time" << endl;
//     for (int i = 0; i < num_elections_to_generate; i++) {
//         auto start_time = chrono::high_resolution_clock::now();
//         auto [res, score] =
//             greedy_dp::farthest_approvalwise_vector(votings_hist,
//             num_voters);
//         auto end_time = chrono::high_resolution_clock::now();
//         auto elapsed_time =
//             chrono::duration_cast<chrono::milliseconds>(end_time -
//             start_time)
//                 .count();
//         cout << i + 3 << ',' << score << ','
//              << double(score) / (num_voters * num_candidates) << ','
//              << elapsed_time / 1000. << endl;

//         // print_vec(res);
//         // cout << endl;
//         votings_hist.push_back(res);
//     }
// }
