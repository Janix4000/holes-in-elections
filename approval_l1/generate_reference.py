import argparse
import os
import pickle
import mapel.elections as mapel
import numpy as np

from scripts.gurobi import gurobi_ilp
from scripts.approvalwise_vector import dump_to_text_file, load_from_text_file


def main():

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--file', type=str,
                        help='File consisting elections', required=True)
    parser.add_argument('--num_instances', type=int, default=12)

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the values from the command line arguments
    filepath = args.file
    num_instances = args.num_instances

    with open(filepath, 'r') as file:
        approvalwise_vectors = load_from_text_file(file)

    last_distance = None
    reference_app_vecs = []
    for i in range(num_instances):
        app_vec, dist = gurobi_ilp(
            approvalwise_vectors, max_dist=last_distance)
        reference_app_vecs.append(app_vec)
        print(f'{i},{dist}')

    dirpath = os.path.dirname(filepath)
    filename = os.path.join(dirpath, 'reference.txt')
    with open(filename, 'w') as file:
        dump_to_text_file(reference_app_vecs, file)


if __name__ == "__main__":
    main()
