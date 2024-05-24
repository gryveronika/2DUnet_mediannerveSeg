import torch
import pickle
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataprocessing import ImageMaskDataset

def compute_class_weights(dataset, num_classes):
    """
    Compute the class weights for a dataset to handle class imbalance.

    Parameters:
    - dataset (list): A list of samples, where each sample is a dictionary with keys 'Image' and 'Mask'.
    - num_classes (int): The number of classes.

    Returns:
    - class_weights (torch.Tensor): A tensor containing the computed class weights.
    """
    class_counts = torch.zeros(num_classes)
    total_samples = 0

    # Iterate through the dataset to count samples for each class
    for sample in dataset:
        mask = sample['Mask']
        for class_idx in range(num_classes):
            class_counts[class_idx] += torch.sum(mask == class_idx)
        total_samples += mask.numel()

    # Compute class weights
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights

# Directory where the dataset is saved
save_dir = "../datasets"

# Load the training dataset from a pickle file
with open(os.path.join(save_dir, "dataset_train.pkl"), "rb") as f:
    dataset_train = pickle.load(f)

# Compute class weights for the dataset
class_weights = compute_class_weights(dataset_train, num_classes=9)
print("Class weights:", class_weights)

# Save the computed class weights to a file
filepath = "../class_weights/class_weights2.pt"
torch.save(class_weights, filepath)