#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

#define all(x) (x).begin(), (x).end()

using vi = vector<int>;
using vvi = vector<vi>;
using vvvi = vector<vvi>;

using voting_hist_t = vector<int>;

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

struct Node {
    int from_y = -1;
    int from_idx = -1;
    int min_dist = 0;
    vector<int> single_dists;
};

struct Cell {
    vector<Node> nodes;
};

pair<voting_hist_t, int> next_voting_hist(
    const vector<voting_hist_t>& votings_hist, const int N) {
    const int R = votings_hist.size();
    const int M = votings_hist.front().size();

    vector<vector<Cell>> cells(M, vector<Cell>(N + 1));

    {
        int x = 0;
        for (int y = N; y >= 0; --y) {
            Node node;
            for (int r = 0; r < R; r++) {
                node.single_dists.push_back(abs(y - votings_hist[r][x]));
            }
            // node.min_dist = *min_element(all(node.single_dists));
            cells[x][y].nodes.push_back(node);
        }
    }
    for (int x = 1; x < M; x++) {
        for (int y = N; y >= 0; --y) {
            const int x_prev = x - 1;
            for (int y_prev = N; y_prev >= y; --y_prev) {
                const auto& prev_cell = cells[x_prev][y_prev];
                for (int idx = 0; idx < (int)prev_cell.nodes.size(); ++idx) {
                    const auto& prev_node = prev_cell.nodes[idx];
                    Node node;
                    node.from_y = y_prev;
                    node.from_idx = idx;
                    for (int r = 0; r < R; r++) {
                        node.single_dists.push_back(
                            abs(y - votings_hist[r][x]) +
                            prev_node.single_dists[r]);
                    }
                    cells[x][y].nodes.push_back(node);
                }
            }
        }
    }

    vector<Node> backs_maxs(N + 1);

    for (int y = 0; y < N + 1; y++) {
        auto& cell = cells.back()[y];
        for (auto& node : cell.nodes) {
            node.min_dist = *min_element(all(node.single_dists));
        }
        auto it =
            max_element(all(cell.nodes), [](const auto& l, const auto& r) {
                return l.min_dist < r.min_dist;
            });
        backs_maxs[y] = *it;
    }

    auto max_el = max_element(
        all(backs_maxs),
        [](const auto& l, const auto& r) { return l.min_dist < r.min_dist; });
    voting_hist_t res(M);
    int y = max_el - backs_maxs.begin();
    Node node = *max_el;

    res.back() = y;
    for (int x = M - 2; x >= 0; x--) {
        y = node.from_y;
        node = cells[x][y].nodes[node.from_idx];
        res[x] = y;
    }

    return {res, max_el->min_dist};
}

signed main(int argc, char** args) {
    if (argc != 3 + 1) {
        cerr << "Usage: " << args[0]
             << " N(voters) M(candidates) R(additional votings)" << endl;
        return 1;
    }
    const int N = stoi(args[1]);
    const int M = stoi(args[2]);
    const int R = stoi(args[3]);
    vector<voting_hist_t> votings_hist = {
        vi(M, N), vi(M, 0), {16, 16, 16, 16, 0, 0, 0, 0}};
    const size_t start_N = votings_hist.size();

    cout << "r,dist,dist_prop,time" << endl;
    for (int i = 0; i < R; i++) {
        auto start_time = chrono::high_resolution_clock::now();
        auto [res, score] = next_voting_hist(votings_hist, N);
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