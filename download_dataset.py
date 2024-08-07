import argparse
import collections
import os
import pickle
import urllib.request

import h5py
import numpy as np
from tqdm import tqdm

from cldt.dataset_infos import DATASET_URLS


def download_dataset_from_url(dataset_url, output_dir):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(output_dir, dataset_name)

    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)

    return dataset_filepath


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def get_dataset(
    dataset_url,
    output_dir,
    env=None,
):
    h5path = download_dataset_from_url(dataset_url, output_dir)

    data_dict = {}
    with h5py.File(h5path, "r") as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ["observations", "actions", "rewards", "terminals"]:
        assert key in data_dict, "Dataset is missing key %s" % key
    N_samples = data_dict["observations"].shape[0]

    if env is not None:
        if env.observation_space.shape is not None:
            assert (
                data_dict["observations"].shape[1:] == env.observation_space.shape
            ), "Observation shape does not match env: %s vs %s" % (
                str(data_dict["observations"].shape[1:]),
                str(env.observation_space.shape),
            )
        assert (
            data_dict["actions"].shape[1:] == env.action_space.shape
        ), "Action shape does not match env: %s vs %s" % (
            str(data_dict["actions"].shape[1:]),
            str(env.action_space.shape),
        )

    if data_dict["rewards"].shape == (N_samples, 1):
        data_dict["rewards"] = data_dict["rewards"][:, 0]
    assert data_dict["rewards"].shape == (N_samples,), "Reward has wrong shape: %s" % (
        str(data_dict["rewards"].shape)
    )
    if data_dict["terminals"].shape == (N_samples, 1):
        data_dict["terminals"] = data_dict["terminals"][:, 0]
    assert data_dict["terminals"].shape == (
        N_samples,
    ), "Terminals has wrong shape: %s" % (str(data_dict["rewards"].shape))
    return data_dict


def process_dataset(data_dict, name, output_dir):
    dataset = data_dict
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == 1000 - 1
        for k in [
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminals",
        ]:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )

    filename = f"{name}.pkl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "wb") as f:
        pickle.dump(paths, f)


def download_dataset(name, output_dir):
    if name not in DATASET_URLS:
        raise ValueError(
            f"Dataset {name} not found in DATASET_URLS (see dataset_infos.py)"
        )

    # create output_dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_url = DATASET_URLS[name]
    data_dict = get_dataset(dataset_url, output_dir=output_dir)
    process_dataset(data_dict, name=name, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
        help="name of the dataset (see dataset_infos.py)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=False,
        default="datasets",
        help="directory to save the dataset",
    )

    args = parser.parse_args()
    download_dataset(args.name, args.output_dir)
