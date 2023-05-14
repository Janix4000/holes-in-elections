#include "brute.hpp"

#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
// #include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;
namespace po = boost::program_options;

signed main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "voters,N", po::value<int>()->required(), "number of voters")(
        "candidates,M", po::value<int>()->required(), "number of candidates")(
        "additional,R", po::value<int>()->required(), "number of new votings")(
        "json", po::value<bool>()->default_value(false), "json output?")(
        "verbose", po::value<int>()->default_value(1), "verbose level");

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

    vector<voting_hist_t> votings_hist = {vi(M, N), vi(M, 0)};
    const size_t start_N = votings_hist.size();

    const int verbose = vm["verbose"].as<int>();

    cout << "r,dist,dist_prop,time" << endl;
    for (int i = 0; i < R; i++) {
        auto start_time = chrono::high_resolution_clock::now();
        auto [res, score] = brute::next_voting_hist(votings_hist, N);
        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time =
            chrono::duration_cast<chrono::milliseconds>(end_time - start_time)
                .count();

        votings_hist.push_back(res);
        if (verbose < 1) continue;
        cout << i + start_N + 1 << ',' << score << ','
             << double(score) / (N * M) << ',' << setprecision(4)
             << double(elapsed_time / 1000.) << endl;
        if (verbose < 2) continue;
        print_vec(res);
        cout << endl;
    }

    if (vm["json"].as<bool>()) {
        json jsonData = votings_hist;
        std::cout << jsonData.dump(4) << std::endl;
    }
}