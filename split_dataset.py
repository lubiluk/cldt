import argparse

from cldt.datasets import split_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=False,
        default='datasets/panda_push_dense.pkl',
        help="path to the dataset",
    )

    parser.add_argument(
        "-ratio",
        "--expert_size_ratio",
        type=float,
        required=False,
        default=0.1,
        help="ratio between expert and random samples",
    )

    parser.add_argument(
        "-s",
        "--number_of_samples",
        type=int,
        required=False,
        default=1000000,
        help="number of samples which be used in training",
    )

    args = parser.parse_args()

    split_dataset(args.dataset, args.expert_size_ratio, args.number_of_samples)
