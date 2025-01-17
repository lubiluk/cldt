import pickle
import random
from cldt.extractors import setup_extractor
from paths import DATA_PATH


def load_dataset(dataset_path):
    dataset = f"{DATA_PATH}/{dataset_path}"
    # Load the dataset
    with open(dataset, "rb") as f:
        dataset = pickle.load(f)

    return dataset


def extract_dataset(dataset, extractor_type, observation_space, **kwargs):
    extractor = setup_extractor(extractor_type, observation_space, **kwargs)
    for i in range(len(dataset)):
        dataset[i]["observations"] = extractor(dataset[i]["observations"])


    observation_space = extractor.observation_space
    return dataset, observation_space


def split_dataset(dataset_path, expert_size_ratio=0.3, number_of_samples=1000000):
    expert_dataset_path = dataset_path.replace(".pkl", "_expert.pkl")
    random_dataset_path = dataset_path.replace(".pkl", "_random.pkl")
    with open(expert_dataset_path, "rb") as f:
        expert_dataset = pickle.load(f)
    with open(random_dataset_path, "rb") as f:
        random_dataset = pickle.load(f)

    expert_size = int(number_of_samples * expert_size_ratio)
    random_size = int(number_of_samples * (1 - expert_size_ratio))
    final_expert_dataset = []
    final_random_dataset = []

    i = 0
    for time_step in expert_dataset:
        if i >= expert_size:
            break
        final_expert_dataset.append(time_step)
        i += len(time_step["rewards"])

    i = 0
    for time_step in random_dataset:
        if i >= random_size:
            break
        final_random_dataset.append(time_step)
        i += len(time_step["rewards"])

    final_dataset = final_expert_dataset + final_random_dataset

    random.shuffle(final_dataset)

    final_dataset_path = dataset_path.replace(
        ".pkl", f"{number_of_samples / 1000}k_expert_ratio_{expert_size_ratio}.pkl"
    )

    with open(final_dataset_path, "wb") as f:
        pickle.dump(final_dataset, f)
