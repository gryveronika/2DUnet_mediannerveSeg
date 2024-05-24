import numpy as np
from torch import save, no_grad, where, tensor
import torch.nn.functional as F

def train_segmentation(dataloader_train, dataloader_validate, epochs, unet, device, optimizer, num_classes, criterion, saveweight_filepath, current_class=None):
    """
    Train a UNet model for segmentation, handling both binary and multiclass cases based on the number of classes.

    Parameters:
    - dataloader_train: DataLoader for the training data.
    - dataloader_validate: DataLoader for the validation data.
    - epochs: Number of training epochs.
    - unet: The UNet model.
    - device: The device to train the model on (e.g., 'cpu' or 'cuda').
    - optimizer: The optimizer to use for training.
    - num_classes: The number of classes (2 for binary, >2 for multiclass).
    - criterion: The loss function.
    - saveweight_filepath: The file path to save the model weights.
    - current_class: The current class for binary classification (only needed if num_classes == 2).

    Returns:
    - trainloss: List of training losses per epoch.
    - validateloss: List of validation losses per epoch.
    """
    print("Start training")

    patience = 20
    counter = 0
    bestloss = float("inf")

    trainloss = []
    validateloss = []
    for epoch in range(epochs):

        losses = []

        unet.to(device)
        unet.train()

        for i, image_batch in enumerate(dataloader_train):
            print("Epoch: ", epoch + 1, " of ", epochs, " Batch: ", i + 1, " of ", len(dataloader_train))

            image_to_segment = image_batch['Image'].to(device)
            ground_truth_mask = image_batch['Mask'].to(device)

            if num_classes == 2:
                if current_class is None:
                    raise ValueError("current_class must be specified for binary classification.")
                mask_current_class = where(ground_truth_mask == current_class, 1, 0)
                target_one_hot = F.one_hot(mask_current_class.long().squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
            else:
                target_one_hot = F.one_hot(ground_truth_mask.long().squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

            model_output = unet(image_to_segment)
            loss = criterion(model_output, target_one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu())

        trainloss.append(np.mean(losses))
        print("TRAINLOSS", trainloss)

        unet.eval()
        losses = []

        for i, image_batch in enumerate(dataloader_validate):
            print("Epoch: ", epoch + 1, " of ", epochs, " Validate batch: ", i + 1, " of ", len(dataloader_validate))

            image_to_segment = image_batch['Image'].to(device)
            ground_truth_mask = image_batch['Mask'].to(device)

            if num_classes == 2:
                mask_current_class = where(ground_truth_mask == current_class, 1, 0)
                target_one_hot = F.one_hot(mask_current_class.long().squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
            else:
                target_one_hot = F.one_hot(ground_truth_mask.long().squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

            model_output = unet(image_to_segment)
            loss = criterion(model_output, target_one_hot)
            losses.append(loss.detach().cpu())

        validateloss.append(np.mean(losses))
        print("VALIDATELOSS", validateloss)

        if validateloss[epoch] < bestloss:
            bestloss = validateloss[epoch]
            save(unet.state_dict(), saveweight_filepath)
            counter = 0
        else:
            counter = counter + 1
            if counter > patience:
                print("Early stopping!")
                return trainloss, validateloss
    return trainloss, validateloss

