import os
import csv
from scipy.stats import entropy
import numpy as np


def pct_cropped_has_bigger_uncertainty(unc_orig, unc_cropped):
    return (unc_orig < unc_cropped).float().mean()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_to_csv(file_path, data):
    """
    Save data to a CSV file. Create the directory and file if they don't exist.
    
    Parameters:
        file_path (str): The path to the CSV file.
        data (dict): A dictionary containing the data to save. Keys should match the column names.
    """
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the column headers
    headers = ['model', 'dataset', 'recall criterion', 'R-AUROC', 'Recall@1', 'pct. cropped image has higher uncertainty']
    
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    
    # Open the file in append mode
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write headers if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data
        writer.writerow(data)

def compute_entropy(seg_patch):
    """
    Compute the entropy of the class distribution in a segmentation patch.
    
    :param seg_patch: 2D numpy array (patch of segmentation mask)
    :return: Entropy value
    """
    unique, counts = np.unique(seg_patch, return_counts=True)
    probs = counts / np.sum(counts)  # Normalize to get probabilities
    return entropy(probs)  # Shannon entropy

def entropy_dispersion(segmentation_mask, patch_size=32):
    """
    Compute entropy-based dispersion over an image by analyzing patches.
    
    :param segmentation_mask: 2D numpy array (segmentation mask)
    :param patch_size: Size of the patches for entropy computation
    :return: Mean entropy of the image
    """
    h, w = segmentation_mask.shape
    entropies = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = segmentation_mask[i:i+patch_size, j:j+patch_size]
            if patch.size == 0:
                continue
            entropies.append(compute_entropy(patch))

    return np.mean(entropies)  # Average entropy as dispersion metric
