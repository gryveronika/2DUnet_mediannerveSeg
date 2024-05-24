import torch
import torch.nn.functional as F
import pickle

from source_code.Unet import unet
from source_code.config import *
from dataprocessing import ImageMaskDataset


"""
Image Segmentation and Result Saving Script

This script loads a pre-trained U-Net model to perform image segmentation on a test dataset.
The segmentation results are saved for later analysis or use.
"""


unet.load_state_dict(torch.load(path_savedweights))
unet.to(device)
unet.eval()

save_dir = 'datasets'
with open(os.path.join(save_dir, f"dataset_testp{test_patient}.pkl"), "rb") as f:
    dataset_test = pickle.load(f)

# save_dir = 'datasets'
# with open(os.path.join(save_dir, f"dataset_test.pkl"), "rb") as f:
#     dataset_test = pickle.load(f)

if __name__ == '__main__':
    predictions = []

    for image in dataset_test:

        image_to_segment = image['Image'].to(device)
        ground_truth_mask = image['Mask'].to(device)

        with torch.no_grad():
            image_to_segment = image_to_segment.unsqueeze(0).to(device)
            model_output = unet(image_to_segment)
            pred_probability_maps = F.softmax(model_output, dim=1)
            prediction = torch.argmax(pred_probability_maps, dim=1)


            predictions.append(prediction.cpu())
            print("Frame nr ",len(predictions), "finish!")

    torch.save(predictions, saveprediction_path)
