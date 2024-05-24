import matplotlib.pyplot as plt
import numpy as np
from torch import load, where
from matplotlib.colors import ListedColormap, Normalize
import pickle
from skimage import segmentation, color
from source_code.config import *
from dataprocessing import ImageMaskDataset


"""
This script visualizes binary and multiclass segmentation results against ground truth labels.
For each image in the test dataset, it loads the image, ground truth segmentation mask, and predicted segmentation masks.
It overlays the segmentation masks on the original images and displays them for visual inspection.
"""

if __name__ == '__main__':
    # Load the test dataset
    save_dir = 'datasets'
    with open(os.path.join(save_dir, f"dataset_testp{test_patient}.pkl"), "rb") as f:
        dataset_test = pickle.load(f)

    # Load predictions from saved files
    predictions = load(pred1_path)
    predictions2 = load(pred2_path)

    # Iterate through each image in the test dataset
    for idx, image in enumerate(dataset_test):
        im = image["Image"]
        seg = image["Mask"]

        pred = predictions[idx]
        pred2 = predictions2[idx]

        pred_multiclass = pred

        # Prepare data for visualization
        seg = where(seg == 1, seg, 0)
        pred = where(pred == 1, pred, 0)
        us_image = np.squeeze(im.numpy())
        us_GT = np.squeeze(seg.numpy())
        us_pred = np.squeeze(pred.numpy())
        us_pred2 = np.squeeze(pred2.numpy())

        ########################################
        # FOR BINARY SEGMENTATION COMPARISON AGAINST GROUND TRUTH

        # Overlay ground truth and predicted segmentation masks on the image
        result1_image = segmentation.mark_boundaries(us_image, us_GT, color=(0, 255, 0), mode='thin')
        result1_image = segmentation.mark_boundaries(result1_image, us_pred, color=(255, 0, 0), mode='thin')
        result1_image = segmentation.mark_boundaries(result1_image, us_pred2, color=(0, 0, 255), mode='thin')

        # Display the image with overlaid masks
        fig1 = plt.imshow(result1_image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        ##################################
        # FOR MULTICLASS SEGMENTATION AGAINST GROUND TRUTH

        # Define custom colors and colormap for visualization
        custom_colors = ["black", "yellow", "red", "blue", "purple", "mediumspringgreen", "white", "orange", "green"]
        custom_cmap = ListedColormap(custom_colors)
        norm = Normalize(vmin=0, vmax=8)

        color_GT = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (128, 0, 128), (0, 238, 144), (255, 255, 255),
                      (247, 152, 29), (6, 148, 49)]
        color_pred = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (128, 0, 128), (0, 238, 144), (255, 255, 255),
                      (247, 152, 29), (6, 148, 49)]

        # Overlay ground truth labels on the image using custom colors
        result2_image = color.label2rgb(us_GT, us_image, colors=color_GT, alpha=0.001, bg_label=0, bg_color=None)
        fig2 = plt.imshow(result2_image, cmap=custom_cmap, interpolation='nearest', norm=norm)
        plt.show()

        # Overlay predicted labels on the image using custom colors
        for label in range(1, int(np.max(us_pred) + 1)):
           temp_us_pred = np.where(us_pred == label, 1, 0)
           temp_us_GT = np.where(us_GT == label, 1, 0)
           result2_image = segmentation.mark_boundaries(result2_image, temp_us_pred, color=color_pred[label - 1],
                                                        mode='thin')
        fig3 = plt.imshow(result2_image)
        plt.show()
