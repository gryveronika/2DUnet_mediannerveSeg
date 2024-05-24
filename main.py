import numpy as np
from torch import optim, manual_seed
import pickle
import torch
from pytictoc import TicToc

from source_code.Unet import unet, count_parameters
from source_code.weights_init import weights_init
from source_code.training import train_segmentation
from source_code.loss_functions import DiceLoss, FocalLoss
from source_code.config import *
from dataprocessing import ImageMaskDataset

"""
Image Segmentation Training 

This script trains a U-Net model for image segmentation using the provided training and validation data.
The training loss and validation loss are computed and saved for analysis.

The trained model weights are saved at `saveweights_filepath`.
The training and validation losses are saved as a numpy array at `saveloss_filepath`.
"""

## Load the data from dataloaders
save_dir = "dataloaders"
with open(os.path.join(save_dir, "dataloader_train.pkl"), "rb") as f:
    dataloader_train = pickle.load(f)
with open(os.path.join(save_dir, "dataloader_val.pkl"), "rb") as f:
    dataloader_validate = pickle.load(f)

if __name__ == '__main__':
    t = TicToc()
    t.tic()


    num_weights = count_parameters(unet)
    print(f"Number of trainable parameters in the UNet model: {num_weights}")

    #class_weights = torch.load("class_weights/class_weights.pt")

    manual_seed(111)
    unet.apply(weights_init)
    unet_opt = optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-3)

    criterion = DiceLoss()

    trainloss, validateloss = train_segmentation(dataloader_train, dataloader_validate, epochs, unet, device, unet_opt,
                                               num_classes=num_classes, criterion=criterion,
                                               saveweight_filepath=saveweights_filepath, current_class=current_class)

    np.save(saveloss_filepath, np.array([trainloss, validateloss]))

    t.toc()

