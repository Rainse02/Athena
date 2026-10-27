import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pandas as pd

from settings import DATA_ROOT, OUTPUT_ROOT

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# The data path
path = str(DATA_ROOT / "segmented" / "InternalCrack")
image_files = sorted([f for f in os.listdir(path) if f.endswith(".tif")])

# The first image
first_image = cv2.imread(os.path.join(path, image_files[0]), cv2.IMREAD_GRAYSCALE)
accumulated_map = np.zeros_like(first_image, dtype=np.float32)

"""
Traverse slices and accumulate them.
For the Internal Crack, the first 500 images are the part including the tab 
"""
for file in image_files[:500]:  # [:500]
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    accumulated_map += img  # 累加

# Obtain the coordinates of the center point (the center of the image)
h, w = accumulated_map.shape
x_c, y_c = w // 2, h // 2

max_r = 480
radial_distribution = np.zeros(max_r + 1, dtype=np.float32)


# Calculate the angle from each pixel to the x-axis
y_indices, x_indices = np.indices(accumulated_map.shape)
r = np.sqrt((x_indices - x_c) ** 2 + (y_indices - y_c) ** 2).astype(np.int32)
# a = (180+np.arctan2(x_indices - x_c, y_indices - y_c)*180/math.pi).astype(np.int32)
# The y-axis is in the opposite direction, take the opposite here
a = (np.arctan2(y_c - y_indices, x_indices - x_c) * 180 / math.pi).astype(np.int32)

for radius in range(max_r + 1):
    radial_distribution[radius] = np.sum(accumulated_map[r == radius])

"""
Collect cumulative values from different perspectives.
Different intervals represent different statistical frequencies
 (every 1 degree, every 2 degrees, every 4 degrees, every 10 degrees)
"""

angle_distribution = np.zeros(360)
for angle in range(-180, 180):
    location = (a == angle) & (200 < r) & (r < 480)
    angle_distribution[angle + 180] = np.sum(accumulated_map[location])

# angle_distribution = np.zeros(180)
# for angle in range(180):
#     angle_distribution[angle] = np.sum(accumulated_map[a == angle]) + np.sum(accumulated_map[a == angle*2+1])

# angle_distribution = np.zeros(90)
# for angle in range(-45, 45):
#     location = ((a == angle*4) | (a == angle*4+1) | (a == angle*4+2) | (a == angle*4+3)) & (0 < r) & (r < 500)
#     angle_distribution[angle] = np.sum(accumulated_map[location])

# angle_distribution = np.zeros(36)
# for angle in range(-18, 18):
#     location = ((a >= angle*10) & (a < (angle + 1)*10)) & (0 < r) & (r < 500)
#     angle_distribution[angle+18] = np.sum(accumulated_map[location])

# Normalize to make the distribution easier to compare
angle_distribution /= np.max(angle_distribution)

# # Save the data
# results = pd.DataFrame({
#     'Angle (degree)': (np.arange(len(angle_distribution)) - 180),
#     'Intensity': angle_distribution,
# })
# with pd.ExcelWriter(OUTPUT_ROOT / "angular_distribution.xlsx",
#                     mode='a', engine='openpyxl') as writer:
#     results.to_excel(writer, sheet_name=f"Angle distribution ALL (1 degree)", index=False)

# Draw a radial distribution curve
plt.figure(figsize=(8, 5))
# plt.plot((4*np.arange(len(angle_distribution)) - 180), angle_distribution, label="Angle Distribution")
# plt.plot((10*np.arange(len(angle_distribution)) - 180), angle_distribution, label="Angle Distribution")
plt.plot(
    (np.arange(len(angle_distribution)) - 180),
    angle_distribution,
    label="Angle Distribution",
)
plt.xlabel("Angle from Positive X-ax (degree)")
plt.ylabel("Normalized Intensity")
plt.title("Angle Distribution")
plt.legend()
plt.show()
