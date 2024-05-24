import torch
from skimage import measure


def keep_largest_regions(predictions):
    """
    Keeps only the largest connected region for each class in the predictions.

    Parameters:
    predictions (torch.Tensor): A tensor of shape (batch_size, num_classes, height, width)
                                containing the predicted segmentation masks.

    Returns:
    torch.Tensor: A tensor of the same shape as `predictions` with only the largest connected
                  region for each class retained.
    """
    cleaned_predictions = torch.zeros_like(predictions)

    for idx in range(predictions.shape[0]):  # Iterate over batch size
        for i in range(1, predictions.shape[1]):  # Iterate over classes
            onehot = predictions[idx][i]

            # Label connected regions
            labeled_onehot = measure.label(onehot.cpu().numpy())
            regions = measure.regionprops(labeled_onehot)

            largest_region = None
            max_area = 0

            # Find the largest region
            for region in regions:
                if region.area > max_area:
                    max_area = region.area
                    largest_region = region

            # Retain the largest connected region
            if largest_region is not None:
                largest_connected_region = torch.zeros_like(onehot)
                largest_connected_region[labeled_onehot == largest_region.label] = 1
                cleaned_predictions[idx][i] = largest_connected_region

    return cleaned_predictions
