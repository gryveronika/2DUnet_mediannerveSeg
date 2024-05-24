import numpy as np
import matplotlib.pyplot as plt

# Load the data
lr_small = np.load("../saved_losses/YOUR_FILE_NAME.npy")
lr_medium = np.load("../saved_losses/YOUR_FILE_NAME.npy")
lr_large = np.load("../saved_losses/YOUR_FILE_NAME.npy")


blue = "#80CDC1"
red = "#F4A582"
green="#33B864"

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Plot training loss on the first subplot
ax[0].plot(lr_large[0], color=blue, label="1e-3")
ax[0].plot(lr_medium[0], color=red, label="1e-4")
ax[0].plot(lr_small[0], color=green, label="1e-5")
ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

# Plot validation loss on the second subplot
ax[1].plot(lr_large[1], color=blue, label="1e-3")
ax[1].plot(lr_medium[1], color=red, label="1e-4")
ax[1].plot(lr_small[1], color=green, label="1e-5")
ax[1].set_title("Validation Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()

plt.tight_layout()
plt.show()