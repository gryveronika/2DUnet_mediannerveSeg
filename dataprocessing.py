import os
from skimage import io, color, segmentation
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import albumentations as A
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from source_code.config import root_dir
import pickle
from matplotlib.colors import ListedColormap, Normalize
import natsort





# Define a dataset class for loading images and corresponding masks
class ImageMaskDataset(Dataset):

    def __init__(self, image_paths, mask_paths, dataset_type='train'):
        """
        Initialize the ImageMaskDataset class.

        Parameters:
            image_paths (list): List of file paths to images.
            mask_paths (list): List of file paths to corresponding masks.
            dataset_type (str): Type of dataset, either 'train' or 'test'. Default is 'train'.
        """
        if dataset_type == 'train':
            # Define transformations for training dataset
            self.transforms = A.ReplayCompose([
                A.Resize(height=416, width=256, interpolation=cv2.INTER_NEAREST, always_apply=True),
                A.HorizontalFlip(p=0.50),
                A.VerticalFlip(p=0.50),
            ])
        elif dataset_type == 'test':
            # Define transformations for testing dataset
            self.transforms = A.ReplayCompose([
                A.Resize(height=416, width=256, interpolation=cv2.INTER_NEAREST, always_apply=True)
            ])

        # Store image and mask paths
        self.img_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.

        Parameters:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing the image and its corresponding mask.
        """
        # Load image and mask
        img = io.imread(self.img_paths[idx], plugin="simpleitk")
        mask = io.imread(self.mask_paths[idx], plugin="simpleitk")

        # Apply transformations to image
        data_image = self.transforms(image=img)
        # Apply transformations to mask
        data_mask = self.transforms.replay(data_image["replay"], image=mask)

        # Get transformed image and mask
        img2 = data_image["image"]
        mask2 = data_mask["image"]

        # Convert data types and apply additional transformations if needed
        mask2 = mask2.astype(np.float32)
        img2 = ToTensor()(img2)
        mask2 = ToTensor()(mask2)

        # Return dictionary containing image and mask
        return {'Image': img2, 'Mask': mask2}


# Flatten a dictionary containing image and mask data into separate lists
def flatten_dict_to_lists(dict_data):
    """
    Flatten a dictionary containing image and mask data into separate lists.

    Parameters:
        dict_data (dict): Dictionary containing image and mask data.

    Returns:
        tuple: Tuple containing two lists, one for images and one for masks.
    """
    images_list = []
    masks_list = []
    for key, value in dict_data.items():
        value = np.array(value)
        images_list.extend(value[0].flatten())  # Flatten images and extend the images list
        masks_list.extend(value[1].flatten())  # Flatten masks and extend the masks list
    return images_list, masks_list


# Flatten image and mask data for a single test individual
def flatten_dict_to_lists_one_test_individ(dict_data, individ):
    """
    Flatten image and mask data for a single test individual.

    Parameters:
        dict_data (dict): Dictionary containing image and mask data.
        individ (str): Identifier for the individual.

    Returns:
        tuple: Tuple containing two lists, one for images and one for masks.
    """
    images_list = []
    masks_list = []
    value = np.array(dict_data[individ])
    images_list.extend(value[0].flatten())  # Flatten images and extend the images list
    masks_list.extend(value[1].flatten())  # Flatten masks and extend the masks list
    return images_list, masks_list


if __name__ == '__main__':
    root_dir = root_dir

    # Collect paths of images and masks for each patient
    all_paths = {}
    for patient_dir in os.listdir(root_dir):
        patient_slices = []
        patient_path = os.path.join(root_dir, patient_dir)
        if os.path.isdir(patient_path):
            for recording_dir in os.listdir(patient_path):
                recording_path = os.path.join(patient_path, recording_dir)
                if os.path.isdir(recording_path):
                    slice_paths = [os.path.join(recording_path, filename) for filename in os.listdir(recording_path) if
                                   filename.endswith(".mhd")]
                    patient_slices.extend(slice_paths)

        # Append patient slices to all_paths
        all_paths[patient_dir] = patient_slices

    # Preprocess data, remove black/wrong segmentations, and organize into a dictionary
    for patient_dir in all_paths.keys():
        print(patient_dir)
        slice_paths_for_patient = all_paths[patient_dir]

        # Divide into mask and image paths
        ending_condition = "_gt.mhd"
        mask_paths_for_patient = [item for item in slice_paths_for_patient if item.endswith(ending_condition)]

        for item in mask_paths_for_patient:
            slice_paths_for_patient.remove(item)

        image_paths_for_patient = slice_paths_for_patient

        # Change the file extension for sorting
        old_ending = ".mhd"
        new_ending = "_im.mhd"
        image_paths_for_patient = [item.replace(old_ending, new_ending) for item in image_paths_for_patient]

        # Sort paths
        image_paths_for_patient = natsort.natsorted(image_paths_for_patient)
        mask_paths_for_patient = natsort.natsorted(mask_paths_for_patient)

        # Change file extensions back to original
        old_ending = "_im.mhd"
        new_ending = ".mhd"
        image_paths_for_patient = [item.replace(old_ending, new_ending) for item in image_paths_for_patient]

        # Remove black/wrong segmentations
        indices_to_remove = []
        a = 0
        for i, item in enumerate(mask_paths_for_patient):
            mask_check = io.imread(item, plugin="simpleitk")
            if np.sum(mask_check) == 0:
                a += 1
                indices_to_remove.append(i)

        # Remove items from both lists
        for index in sorted(indices_to_remove, reverse=True):
            del mask_paths_for_patient[index]
            del image_paths_for_patient[index]
        print("Removed black", a)

        # Add back as a 2D array with [[images], [masks]]
        all_paths[patient_dir] = [image_paths_for_patient, mask_paths_for_patient]

    # Randomly choose train, validation and test set the first time

    # indices = list(all_paths.keys())
    # random.seed(222)
    # random.shuffle(indices)
    #
    # # Calculate the index where 90% of the data ends (for training set)
    # train_size = int(0.8 * len(indices))
    #
    # # Split the shuffled indices into training and test indices
    # train_indices = indices[:train_size]
    # test_indices = indices[train_size:]
    #
    # # Use the indices to split your dictionary of patients into training and test sets
    # train_patients = {index: all_paths[index] for index in train_indices}
    # test_patients = {index: all_paths[index] for index in test_indices}
    #
    # # Shuffle the indices of train_patients
    # train_indices = list(train_patients.keys())
    # random.seed(111)
    # random.shuffle(train_indices)
    #
    # # Calculate the index where 80% of the data ends (for training set)
    # train_size = int(0.8 * len(train_indices))
    #
    # # Split the shuffled indices into training and validation indices
    # train_indices_split = train_indices[:train_size]
    # val_indices = train_indices[train_size:]


    # Use the randomly chosen indices for train, validation, and test sets. To have the same test and train patients they must be defined,
    #since using the same seed locally and reemotely will give different splits
    train_indices_split = ['003', '007', '023', '035', '044', '034', '051', '037', '042', '022', '039', '012', '014',
                           '004', '050', '013', '010', '024', '027', '015', '020', '021', '032', '002', '011', '016',
                           '001', '026', '009', '048']
    val_indices = ['018', '049', '043', '030', '008', '025', '017', '031']
    test_indices = ['028', '006', '045', '033', '005', '036', '041', '019', '029', '052']

    # Define train, validation, and test datasets
    train_patients_final = {index: all_paths[index] for index in train_indices_split}
    val_patients = {index: all_paths[index] for index in val_indices}
    test_patients = {index: all_paths[index] for index in test_indices}

    # Convert training data
    train_images, train_masks = flatten_dict_to_lists(train_patients_final)

    # Shuffle train data
    combined = list(zip(train_images, train_masks))
    random.seed(102)
    random.shuffle(combined)
    train_images, train_masks = zip(*combined)

    # Convert validation and test data
    val_images, val_masks = flatten_dict_to_lists(val_patients)
    test_images, test_masks = flatten_dict_to_lists(test_patients)

    # Make datasets and dataloaders for train, validation, and test sets
    dataset_train = ImageMaskDataset(image_paths=train_images, mask_paths=train_masks, dataset_type='train')
    dataloader_train = DataLoader(dataset_train, batch_size=16, num_workers=2, drop_last=True)

    dataset_validate = ImageMaskDataset(image_paths=val_images, mask_paths=val_masks, dataset_type='test')
    dataloader_validate = DataLoader(dataset_validate, batch_size=16, num_workers=2, drop_last=True)

    dataset_test = ImageMaskDataset(image_paths=test_images, mask_paths=test_masks, dataset_type='test')
    dataloader_test = DataLoader(dataset_test, batch_size=16, num_workers=2, drop_last=True)

    # Save the datasets and dataloaders using pickle
    save_dir = "dataloaders"
    with open(os.path.join(save_dir, "dataloader_train.pkl"), "wb") as f:
        pickle.dump(dataloader_train, f)

    with open(os.path.join(save_dir, "dataloader_val.pkl"), "wb") as f:
        pickle.dump(dataloader_validate, f)

    with open(os.path.join(save_dir, "dataloader_test.pkl"), "wb") as f:
        pickle.dump(dataloader_test, f)

    save_dir = "datasets"
    with open(os.path.join(save_dir, "dataset_train.pkl"), "wb") as f:
        pickle.dump(dataset_train, f)

    with open(os.path.join(save_dir, "dataset_val.pkl"), "wb") as f:
        pickle.dump(dataset_validate, f)

    with open(os.path.join(save_dir, "dataset_test.pkl"), "wb") as f:
        pickle.dump(dataset_test, f)


    for i,index in enumerate(test_indices):
        # Flatten image and mask data for each patient
        test_images, test_masks = flatten_dict_to_lists_one_test_individ(test_patients, index)

        # Create dataset for each patient
        dataset_test_p = ImageMaskDataset(image_paths=test_images, mask_paths=test_masks, dataset_type='test')

        # Save the dataset
        with open(os.path.join(save_dir, f"dataset_testp{i+1}.pkl"), "wb") as f:
            pickle.dump(dataset_test_p, f)


