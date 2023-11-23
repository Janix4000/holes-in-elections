from scripts.bindings import greedy_dp, pairs


if __name__ == "__main__":
    from scripts.approvalwise_vector import ApprovalwiseVector

    num_candidates = 10
    num_voters = 100
    num_instances = 20

    approvalwise_vectors = [
        ApprovalwiseVector([num_voters] * num_candidates, num_voters),
        ApprovalwiseVector([0] * num_candidates, num_voters),
    ]

    for _ in range(num_instances):
        print("Running greedy_dp...")
        print("Running pairs...")
        new_vec, dist = greedy_dp(approvalwise_vectors)
        print((new_vec, dist))

        print(pairs(approvalwise_vectors))

        approvalwise_vectors.append(new_vec)
