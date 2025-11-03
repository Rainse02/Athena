import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.bipartite.basic import color
# from imageio.v2 import sizes

plt.rc('font',family='Arial', size=15)
path = r"E:\Work\Electrode_fault_identification\Results_Individual_510-2100\InternalCrack"
image_files = sorted([f for f in os.listdir(path) if f.endswith(".tif")])

# Loa the first image and initialize the accumulation matrix
first_image = cv2.imread(os.path.join(path, image_files[0]), cv2.IMREAD_GRAYSCALE)
accumulated_map = np.zeros_like(first_image, dtype=np.float32)

size = 0.016849769150315919 #mm

# Traverse all slices and accumulate them
for file in image_files:
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    accumulated_map += img  # Summation

# Obtain the coordinates of the center point (the center of the image)
h, w = accumulated_map.shape
x_c, y_c = w // 2, h // 2

# Calculate the distance from each pixel to the center
y_indices, x_indices = np.indices(accumulated_map.shape)
r = np.sqrt((x_indices - x_c)**2 + (y_indices - y_c)**2).astype(np.int32)

# Calculate the cumulative value under different radii
# max_r = int(np.max(r))
max_r = 475
radial_distribution = np.zeros(max_r + 1, dtype=np.float32)

for radius in range(max_r + 1)[1:]:
        radial_distribution[radius] = (np.sum(accumulated_map[r == radius])) / (radius**2)

print(np.sum(radial_distribution))
print(np.sum(accumulated_map))
# Normalize to make the distribution easier to compare
radial_distribution /= np.max(radial_distribution)


# # Save the data
# results = pd.DataFrame({
#     'R from center (pixel)': np.arange(max_r + 1),
#     'R from center (mm)': np.arange(max_r + 1)*size,
#     'Normalized Density': radial_distribution.squeeze()
# })
# with pd.ExcelWriter(f"E:\Work\Electrode_fault_identification\Count\Internal Crack\Internal count.xlsx",
#                     mode='a', engine='openpyxl') as writer:
#     results.to_excel(writer, sheet_name=f"Radial distribution Surface", index=False)
#
# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(np.arange(max_r + 1)*size, radial_distribution, label="Internal", color='#D64F38')
plt.xlabel("Distance from Center (mm)")
plt.ylabel("Normalized Density")
# plt.title("Radial Distribution")
# plt.legend()
plt.grid(True)
plt.savefig(r"E:\Work\Electrode_fault_identification\Count\Internal Crack\Distance from Center.svg")
plt.show()

