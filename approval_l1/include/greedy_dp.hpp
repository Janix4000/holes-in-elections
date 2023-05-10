#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "definitions.hpp"

namespace greedy_dp {
pair<voting_hist_t, int> next_voting_hist(
    const vector<voting_hist_t>& votings_hist, const int N) {
    const int R = votings_hist.size();
    const int M = votings_hist.front().size();
    const int INF = N * M + 1;

    vvvi dp_per_voting(M, vvi(N + 1, vi(R, 0)));
    vvi dp(M, vi(N + 1, 0));
    vvi from(M, vi(N + 1, -1));

    for (int y = N; y >= 0; --y) {
        const int x = 0;
        for (int r = 0; r < R; r++) {
            dp_per_voting[x][y][r] = abs(y - votings_hist[r][x]);
        }
        dp[x][y] = *min_element(all(dp_per_voting[x][y]));
    }

    for (int x = 1; x < M; x++) {
        for (int y = N; y >= 0; --y) {
            int max_y = -1;
            int max_for_y = -1;
            const int x_prev = x - 1;
            for (int y_prev = N; y_prev >= y; y_prev--) {
                int min_for_r = INF;
                for (int r = 0; r < R; r++) {
                    const int val = abs(y - votings_hist[r][x]) +
                                    dp_per_voting[x_prev][y_prev][r];
                    min_for_r = min(min_for_r, val);
                }
                if (min_for_r > max_for_y) {
                    max_for_y = min_for_r;
                    max_y = y_prev;
                }
            }

            for (int r = 0; r < R; r++) {
                dp_per_voting[x][y][r] = abs(y - votings_hist[r][x]) +
                                         dp_per_voting[x_prev][max_y][r];
            }

            dp[x][y] = *min_element(all(dp_per_voting[x][y]));
            from[x][y] = max_y;
        }
    }

    voting_hist_t res(M);
    auto max_el = max_element(all(dp.back()));
    int y = max_el - dp.back().begin();
    res.back() = y;
    for (int x = M - 2; x >= 0; x--) {
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