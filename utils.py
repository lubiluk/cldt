import pickle
import random
from random import sample


def split_dataset(dataset_path, expert_size_ratio=0.3, number_of_samples=None):
    expert_dataset_path = dataset_path.replace(".pkl", "_expert.pkl")
    random_dataset_path = dataset_path.replace(".pkl", "_random.pkl")
    with open(expert_dataset_path, "rb") as f:
        expert_dataset = pickle.load(f)
    with open(random_dataset_path, "rb") as f:
        random_dataset = pickle.load(f)

    expert_size = int(len(expert_dataset) * expert_size_ratio)
    random_size = int(len(random_dataset) * (1 - expert_size_ratio))
    random_dataset = sample(random_dataset, random_size)
    expert_dataset = sample(expert_dataset, expert_size)
    final_dataset = expert_dataset + random_dataset
    final_dataset = final_dataset[:number_of_samples] if number_of_samples is not None else final_dataset

    random.shuffle(final_dataset)

    return final_dataset
