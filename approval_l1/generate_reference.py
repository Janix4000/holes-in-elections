import argparse
import os
import time

import pandas as pd

from scripts.gurobi import gurobi_ilp
from scripts.approvalwise_vector import dump_to_text_file, load_from_text_file


def main():

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--file', type=str,
                        help='File consisting elections', required=True)
    parser.add_argument('--num_instances', type=int, default=12)
    args = parser.parse_args()

    filepath = args.file
    num_instances = args.num_instances

    with open(filepath, 'r') as file:
        approvalwise_vectors = list(load_from_text_file(file).values())

    dirpath = os.path.dirname(filepath)
    ref_filepath = os.path.join(dirpath, 'reference.txt')

    report_rows = []

    last_distance = None
    reference_app_vecs = []
    report_filepath = os.path.join(dirpath, 'reference_report.csv')
    for i in range(num_instances):
        start_time = time.process_time()
        app_vec, dist = gurobi_ilp(
            approvalwise_vectors, max_dist=last_distance)
        dt = time.process_time() - start_time
        reference_app_vecs.append(app_vec)
        approvalwise_vectors.append(app_vec)
        with open(ref_filepath, 'w') as file:
            dump_to_text_file(reference_app_vecs, file)
        print(f'{i},{dist},{dt}')
        report_rows.append([i, dist, dt])
        pd.DataFrame(report_rows, columns=['i', 'distance', 'dt']).to_csv(
            report_filepath, index=False)


if __name__ == "__main__":
    main()
