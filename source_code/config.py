import os
from torch import device, cuda

# Configuration for GPU usage
GPU_NUM = 1
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = f"{GPU_NUM}"

# Setting the device for PyTorch
device = device("cuda" if cuda.is_available() else "cpu")

# Data root directory
root_dir = "/Users/YOUR_DATA_PATH"

# Model and Training Configuration
multiclass = True
Unet_levels = 5
epochs = 5
batchsize = 16
num_classes = 9  # Number of classes for segmentation
current_class=1  # The class to segment (only for binary segmentation)

# Data and Testing Configuration
test_patient = 9
all_patients_list = [
    "dataset_testp1.pkl", "dataset_testp2.pkl", "dataset_testp3.pkl", "dataset_testp4.pkl",
    "dataset_testp5.pkl", "dataset_testp6.pkl", "dataset_testp7.pkl", "dataset_testp8.pkl",
    "dataset_testp9.pkl", "dataset_testp10.pkl"
]  # List of all patient datasets

# Paths for loading saved model weights and predictions
path_savedweights = 'saved_weights/model_weights.pt'  # Path to load saved model weights from
load_prediction_path = f"Results/model_predictions.pt"  # Path to load predictions from

# Paths for saving predictions and weights
saveprediction_path = f"Results/model_predictions.pt"  # Path to save predictions to
saveweights_filepath = "saved_weights/model_weights.pt"  # Path to save model weights to
saveloss_filepath = "saved_losses/model_losses.npy"  # Path to save training loss values


#Loading segmentations to compare
pred1_path = "Results/model_predictions.pt"
pred2_path = "Results/model_predictions2.pt"
