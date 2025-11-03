import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pandas as pd
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

path = r'E:\Work\Electrode_fault_identification\Count\Surface Crack\Color_resize\SurfaceCrack in_resize_color.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 1, 1, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

result = np.sum(img, axis=0).astype(np.float64)

size = 0.016849769150315919 #mm

# save data
result = pd.DataFrame({
    'Distance from Outside(pixel)': np.arange(len(result)),
    'Distance from Outside(mm)': np.arange(len(result))*size,
    'Detach in — Intensity': result
})

with pd.ExcelWriter(f"E:\Work\Electrode_fault_identification\Count\Surface count.xlsx",
                    mode='a', engine='openpyxl') as writer:
    result.to_excel(writer, sheet_name=f"Surface in", index=False)


# plt.figure(1)
# plt.plot(np.arange(len(result)), result+result2, linewidth=0.1, label='Detach')#

# plt.legend()
# plt.figure(1)
# plt.plot(np.arange(len(result)), result, linewidth=0.5, label='Detach out')#
# plt.legend()
# plt.show()