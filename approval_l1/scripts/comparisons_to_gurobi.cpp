#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

#include "approvalwise_vector.hpp"
#include "definitions.hpp"
#include "greedy_dp.hpp"
#include "pairs.hpp"

using algorithm_t = std::function<std::pair<approvalwise_vector_t, int>(
    const std::vector<approvalwise_vector_t>&, const int)>;

void run_experiment(
    const std::vector<approvalwise_vector_t>& approvalwise_vectors,
    const std::vector<approvalwise_vector_t>& reference_approvalwise_vectors,
    const size_t num_voters, algorithm_t algorithm, std::ostream& report_out,
    std::optional<std::string> output_dir) {
    size_t num_elections_reference = reference_approvalwise_vectors.size();

    report_out
        << "experiment_size,distance,execution_time,num_starting_elections"
        << std::endl;
    for (int num_starting_elections = 0;
         num_starting_elections < num_elections_reference;
         num_starting_elections++) {
        std::vector<approvalwise_vector_t> starting_approvalwise_vectors =
            approvalwise_vectors;
        for (int i = 0; i < num_starting_elections; i++) {
            starting_approvalwise_vectors.push_back(
                reference_approvalwise_vectors[i]);
        }
        std::vector<approvalwise_vector_t> new_approvalwise_vectors;
        for (size_t idx = 0;
             idx < num_elections_reference - num_starting_elections; idx++) {
            auto start_time = std::chrono::system_clock::now();
            auto [farthest_approvalwise_vectors, distance] =
                algorithm(starting_approvalwise_vectors, num_voters);
            float execution_time_s =
                std::chrono::duration_cast<std::chrono::duration<float>>(
                    std::chrono::system_clock::now() - start_time)
                    .count();

            report_out << starting_approvalwise_vectors.size() << ","
                       << distance << "," << execution_time_s << ","
                       << num_starting_elections << std::endl;

            starting_approvalwise_vectors.push_back(
                farthest_approvalwise_vectors);
            new_approvalwise_vectors.push_back(farthest_approvalwise_vectors);
        }

        if (output_dir.has_value()) {
            std::ofstream out(output_dir.value() + "new_approvalwise_vectors_" +
                              std::to_string(num_starting_elections) + ".txt");
            save_approvalwise_vectors(out, new_approvalwise_vectors,
                                      num_voters);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3 + 1) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <reference_input_file> <algorithm> "
                     "[output_dir]\n";
        return 1;
    }
    std::ifstream in(argv[1]);
    auto [approvalwise_vectors, num_voters] = load_approvalwise_vectors(in);
    size_t num_candidates = approvalwise_vectors.front().size();
    size_t num_elections = approvalwise_vectors.size();

    std::ifstream in_ref(argv[2]);
    auto [reference_approvalwise_vectors, _num_voters] =
        load_approvalwise_vectors(in_ref);

    algorithm_t algorithm;

    std::string algorithm_name = argv[3];
    if (algorithm_name == "greedy_dp") {
        algorithm = greedy_dp::farthest_approvalwise_vector;
    } else if (algorithm_name == "pairs") {
        algorithm = pairs::farthest_approvalwise_vector;
    } else {
        std::cerr << "Unknown algorithm: " << algorithm_name << "\n";
        return 1;
    }

    if (argc == 4 + 1) {
        const std::string output_dir =
            std::string(argv[4]) + "/" + algorithm_name + "/";
        std::filesystem::create_directories(output_dir);
        std::ofstream report_out(output_dir + "report.csv",
                                 std::ios::out | std::ios::trunc);
        run_experiment(approvalwise_vectors, reference_approvalwise_vectors,
                       num_voters, algorithm, report_out, output_dir);
    } else {
        run_experiment(approvalwise_vectors, reference_approvalwise_vectors,
                       num_voters, algorithm, std::cout, std::nullopt);
    }

    return 0;
}