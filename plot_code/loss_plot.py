import numpy as np
import matplotlib.pyplot as plt

# Load the data
l5_all = np.load("../saved_losses/YOUR_FILE_NAME.npy")
l5_med = np.load("../saved_losses/YOUR_FILE_NAME.npy")
l6_all = np.load("../saved_losses/YOUR_FILE_NAME.npy")
l6_med = np.load("../saved_losses/YOUR_FILE_NAME.npy")

blue = "#80CDC1"
red = "#F4A582"

# Create subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Plot on each subplot
ax[1, 0].plot(l5_all[0], color=blue, label="Train")
ax[1, 0].plot(l5_all[1], color=red, label="Validate")
ax[1, 0].set_title("Multiclass segmentation with 5 U-Net levels")
ax[1, 0].set_xlabel("Epochs")
ax[1, 0].set_ylabel("Dice loss")
ax[1, 0].set_ylim(0,0.95)
ax[1, 0].legend()

ax[0, 0].plot(l5_med[0], color=blue, label="Train")
ax[0, 0].plot(l5_med[1], color=red, label="Validate")
ax[0, 0].set_title("Binary segmentation with 5 U-Net levels")
ax[0, 0].set_xlabel("Epochs")
ax[0, 0].set_ylabel("Dice loss")
ax[0, 0].set_ylim(0,0.95)
ax[0, 0].legend()

ax[1, 1].plot(l6_all[0], color=blue, label="Train")
ax[1, 1].plot(l6_all[1], color=red, label="Validate")
ax[1, 1].set_title("Multiclass segmentation with 6 U-Net levels")
ax[1, 1].set_xlabel("Epochs")
ax[1, 1].set_ylabel("Dice loss")
ax[1, 1].set_ylim(0,0.95)
ax[1, 1].legend()

ax[0, 1].plot(l6_med[0], color=blue, label="Train")
ax[0, 1].plot(l6_med[1], color=red, label="Validate")
ax[0, 1].set_title("Binary segmentation with 6 U-Net levels")
ax[0, 1].set_xlabel("Epochs")
ax[0, 1].set_ylabel("Dice loss")
ax[0, 1].set_ylim(0,0.95)
ax[0, 1].legend()

plt.tight_layout()
plt.show()

