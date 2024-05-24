import os
import pickle
import torch
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataprocessing import ImageMaskDataset


def plot_histogram(dictionary):
    """
    Plots a histogram from a dictionary where keys are the categories and values are their frequencies.

    Parameters:
    dictionary (dict): A dictionary with categories as keys and frequencies as values.
    """
    # Extract keys and values from the dictionary
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # Plot the histogram
    plt.bar(keys, values, color="#6699CC")
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(rotation=32, fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# Directory where the dataset is saved
save_dir = '../datasets'

# Load the test dataset from a pickle file
with open(os.path.join(save_dir, "dataset_train.pkl"), "rb") as f:
    dataset_test = pickle.load(f)

# List of anatomical structures present in the dataset
structures = ["Background", "Median nerve", "Radial artery", "Pronator quad.", "Flexor carpi rad.", "Radius", "Ulna", "Pisiform", "Scaphoid"]

# Initialize a dictionary to count the occurrences of each structure
count_dict = {structure: 0 for structure in structures}

# Iterate through each sample in the test dataset
for sample in dataset_test:
    mask = sample['Mask']  # Ground truth mask for the structures

    # Check for the presence of each structure in the mask and update the count
    for i, structure in enumerate(structures):
        is_present = torch.any(torch.eq(mask, i))
        if is_present:
            count_dict[structure] += 1

# Print the dictionary containing the count of each structure
print(count_dict)
plot_histogram(count_dict)