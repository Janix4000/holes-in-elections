#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "approvalwise_vector.hpp"
#include "brute.hpp"
#include "greedy_dp.hpp"

using json = nlohmann::json;
namespace po = boost::program_options;

signed main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "voters,N", po::value<int>()->required(), "number of voters")(
        "candidates,M", po::value<int>()->required(), "number of candidates")(
        "additional,R", po::value<int>()->required(), "number of new votings")(
        "algorithm,A", po::value<std::string>()->default_value("brute"),
        "chosen algorithm <brute|greedy_dp>")(
        "json", po::value<bool>()->default_value(false), "json output?")(
        "verbose", po::value<int>()->default_value(1), "verbose level")(
        "load-hists", po::value<std::string>(), "path to json file with hists");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    const int N = vm["voters"].as<int>();
    const int M = vm["candidates"].as<int>();
    const int R = vm["additional"].as<int>();
    const int verbose = vm["verbose"].as<int>();
    const bool json_output = vm["json"].as<bool>();
    const std::string algorithm = vm["algorithm"].as<std::string>();

    if (algorithm != "brute" && algorithm != "greedy_dp") {
        std::cerr << "Error: unknown algorithm" << std::endl;
        return 1;
    }

    vector<approvalwise_vector_t> votings_hist;
    if (vm.count("load-hists")) {
        std::ifstream ifs(vm["load-hists"].as<std::string>());
        json jsonData = json::parse(ifs);
        votings_hist = jsonData.get<vector<approvalwise_vector_t>>();
    } else {
        votings_hist = {vi(M, N), vi(M, 0)};
    }
    const size_t start_N = votings_hist.size();

    if (verbose >= 1) cout << "r,dist,dist_prop,time" << endl;
    if (json_output) {
        cout << "[\n";
        for (const auto& voting : votings_hist) {
            cout << "\t";
            print_vec(voting);
            cout << ",";
            cout << endl;
        }
    }

    for (int i = 0; i < R; i++) {
        auto start_time = chrono::high_resolution_clock::now();
        std::pair<approvalwise_vector_t, int> res;
        if (algorithm == "brute") {
            res = brute::farthest_approvalwise_vector(votings_hist, N);
        } else if (algorithm == "greedy_dp") {
            res = greedy_dp::farthest_approvalwise_vector(votings_hist, N);
        }
        auto [next_voting_hist, score] =
            brute::farthest_approvalwise_vector(votings_hist, N);

        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time =
            chrono::duration_cast<chrono::milliseconds>(end_time - start_time)
                .count();

        votings_hist.push_back(next_voting_hist);
        if (verbose >= 1)
            cout << i + start_N + 1 << ',' << score << ','
                 << double(score) / (N * M) << ',' << setprecision(4)
                 << double(elapsed_time / 1000.) << endl;
        if (json_output) {
            cout << "\t";
            print_vec(next_voting_hist);
            if (i != R - 1) cout << ",";
            cout << endl;
        }
    }

    if (json_output) {
        cout << "]";
    }
}