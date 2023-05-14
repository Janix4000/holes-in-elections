#include "brute.hpp"

#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace po = boost::program_options;

signed main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        // "input", po::value<std::string>(), "input file")(
        // "output", po::value<std::string>(), "output file")(
        "voters,N", po::value<int>(), "number of voters")(
        "candidates,M", po::value<int>(), "number of candidates")(
        "additional,R", po::value<int>(), "number of new votings");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    const int N = vm["voters"].as<int>();
    const int M = vm["candidates"].as<int>();
    const int R = vm["additional"].as<int>();

    // const int N = stoi(argv[1]);
    // const int M = stoi(argv[2]);
    // const int R = stoi(argv[3]);
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