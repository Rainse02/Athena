import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font',family='Arial', size=15)
# Fluid collection results path
path2 = r'E:\Work\Electrode_fault_identification\flatten\thin_result_circle_r2_newslice.png'
# Slice results path
path = r"E:\Work\Electrode_fault_identification\Results_Individual_510-2100\Detach"
image_files = sorted([f for f in os.listdir(path) if f.endswith(".tif")])

# Load the first image and initialize the accumulation matrix
first_image = cv2.imread(os.path.join(path, image_files[0]), cv2.IMREAD_GRAYSCALE)
accumulated_Vertical = []

for file in image_files:
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    accumulated_Vertical.append(img.sum())  # summation

# Normalize to make the distribution easier to compare
accumulated_Vertical /= np.max(accumulated_Vertical)


img = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
"""
The first collection fluid result had excess edges,
which are no longer present in the new version
"""
# result = (np.sum(img, axis=1)[:-1]).astype(np.float64)
result = (np.sum(img, axis=1)).astype(np.float64)
a = np.max(result)
result /= np.max(result)
# result = [result, [0.0]]
result = np.append(result, 0.0)
size = 0.016849769150315919 #mm

# # Save the data
# results = pd.DataFrame({
#     'Distance from top (pixel)': np.arange(len(result)),
#     'Distance from top (mm)': np.arange(len(result))*size,
#     'Current collector Intersection Normalized Intensity': result,
#     'Detach Normalized Intensity': accumulated_Vertical
# })
# with pd.ExcelWriter(f"E:\Work\Electrode_fault_identification\Count\Internal Crack\Internal count.xlsx",
#                     mode='a', engine='openpyxl') as writer:
#     results.to_excel(writer, sheet_name=f"Distance from Electrode Top", index=False)


def create_smooth_gradient_band(data, height=1.0):
    fig, ax = plt.subplots(figsize=(10, height))

    X, Y = np.meshgrid(np.arange(len(data)), np.linspace(0, height, 100))
    Z = np.tile(data, (100, 1))

    im = ax.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
    ax.set_xticks([])
    ax.set_yticks([])

    char = plt.colorbar(im, ax=ax)

    char.outline.set_visible(False)  # Remove the black border
    char.ax.tick_params(left=False, right=False, labelleft=False, labelright=False)  # 去掉刻度和标签
    char.set_label("")  # Do not display label
    # Remove the coordinate axis border
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig
fig = create_smooth_gradient_band(accumulated_Vertical)
plt.show()

# Draw a radial distribution curve
plt.figure(figsize=(12, 5))
plt.plot( np.arange(len(accumulated_Vertical))*size, accumulated_Vertical, label="Detach", color='#D64F38')
plt.plot(np.arange(len(result))*size, result, label="Current Collector Intersection", color='#699AB3')
plt.xlabel("Distance from Electrode Top (mm)")
plt.ylabel("Normalized Intensity")
# plt.title("Vertical Distribution")
# plt.legend(frameon=False)
plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    ncol=2,
    frameon=False,
    fontsize=15
)
plt.grid(True)
plt.savefig(r"E:\Work\Electrode_fault_identification\Count\Internal Crack\Distance from Electrode Top.svg")
plt.show()