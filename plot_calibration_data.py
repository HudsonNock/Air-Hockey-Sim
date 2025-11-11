import numpy as np
import matplotlib.pyplot as plt

# Load the data
calibration_err_train = np.load('calibration_err_train.npy')
calibration_err_test = np.load('calibration_err_test.npy')

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Define bins from 0 to 3 mm with 0.2 mm width
bins = np.arange(-3.2, 3.2, 0.2)

# Plot histogram for training errors
axes[0].hist(calibration_err_train, bins=bins, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Calibration Error (mm)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Training Calibration Errors')
axes[0].grid(True, alpha=0.3)

# Plot histogram for test errors - X component
axes[1].hist(calibration_err_test[:, 0], bins=bins, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Calibration Error (mm)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Test Calibration Errors - X')
axes[1].grid(True, alpha=0.3)

# Plot histogram for test errors - Y component
axes[2].hist(calibration_err_test[:, 1], bins=bins, edgecolor='black', alpha=0.7, color='green')
axes[2].set_xlabel('Calibration Error (mm)')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Test Calibration Errors - Y')
axes[2].grid(True, alpha=0.3)

std_x = np.std(calibration_err_test[:, 0])
std_y = np.std(calibration_err_test[:, 1])

print(f"Standard deviation of test X errors: {std_x:.4f} mm")
print(f"Standard deviation of test Y errors: {std_y:.4f} mm")

plt.tight_layout()
plt.show()