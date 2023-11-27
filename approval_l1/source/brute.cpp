#include "approvalwise_vector.hpp"
#include "definitions.hpp"

namespace brute {

namespace {
struct Node {
    int from_y = -1;
    int from_idx = -1;
    int min_dist = 0;
    std::vector<int> single_dists;

    bool operator<=(const Node& rhs) const {
        if (rhs.single_dists.empty()) return true;
        if (single_dists.empty()) return false;
        for (size_t i = 0; i < single_dists.size(); i++) {
            if (single_dists[i] > rhs.single_dists[i]) return false;
        }
        return true;
    }
};

struct Cell {
    std::vector<Node> nodes;
};
}  // namespace

void remove_non_maximal(std::vector<Node>& nodes) {}

pair<approvalwise_vector_t, int> farthest_approvalwise_vector(
    const std::vector<approvalwise_vector_t>& votings_hist, const int N) {
    const int R = votings_hist.size();
    const int M = votings_hist.front().size();

    std::vector<std::vector<Cell>> cells(M, std::vector<Cell>(N + 1));

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

            auto& cell = cells[x][y];

            // remove_if(all(cell.nodes), )

            std::vector<Node> maximal_nodes;
            copy_if(all(cell.nodes), back_inserter(maximal_nodes),
                    [&cell](const auto& node) {
                        return none_of(
                            all(cell.nodes), [&node](const auto& other_node) {
                                if (other_node.single_dists.size() !=
                                    node.single_dists.size()) {
                                    return false;
                                }
                                bool all_greater = true;
                                for (size_t i = 0; i < node.single_dists.size();
                                     i++) {
                                    all_greater &= other_node.single_dists[i] >
                                                   node.single_dists[i];
                                }
                                return all_greater;
                            });
                    });
            cell.nodes = maximal_nodes;
        }
    }

    std::vector<Node> backs_maxs(N + 1);

    for (int y = 0; y < N + 1; y++) {
        auto& cell = cells.back()[y];
        for (auto& node : cell.nodes) {
            node.min_dist = *std::min_element(all(node.single_dists));
        }
        auto it =
            std::max_element(all(cell.nodes), [](const auto& l, const auto& r) {
                return l.min_dist < r.min_dist;
            });
        backs_maxs[y] = *it;
    }

    auto max_el = max_element(
        all(backs_maxs),
        [](const auto& l, const auto& r) { return l.min_dist < r.min_dist; });
    approvalwise_vector_t res(M);
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
}  // namespace brute
