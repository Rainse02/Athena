import cv2
import numpy as np

from settings import DATA_ROOT, OUTPUT_ROOT

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# The original flatten image
path = str(DATA_ROOT / "images" / "collector_source.tif")
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# Zoom to actual size
img = cv2.resize(img, [13660, 1590])
cv2.imwrite(str(OUTPUT_ROOT / "collector_resized.tif"), img)

# Skeletonization
_, binary = cv2.threshold(
    img, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)  # + cv2.THRESH_OTSU
binary = binary.astype(np.uint8)
# Extract fine lines
img = cv2.ximgproc.thinning(binary)
cv2.imwrite(str(OUTPUT_ROOT / "collector_skeleton.tif"), img)


# Convolution operation
kernel = np.array(
    [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32
)

conv = cv2.filter2D(img.astype(np.float32), -1, kernel)
conv_abs = cv2.normalize(np.abs(conv), None, 0, 255, cv2.NORM_MINMAX)
conv_abs = conv_abs.astype(np.uint8)
# Adaptive threshold binarization
_, binary = cv2.threshold(conv_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(str(OUTPUT_ROOT / "collector_convolution.png"), binary)

# Morphological manipulation enhances connectivity
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Filter small areas
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 15  # Minimum area=1% of image height
# cv2.imshow("process", processed)
for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        cv2.drawContours(processed, [cnt], -1, 0, -1)
result = np.zeros(img.shape)

# # Rectangle
# for cnt in contours:
#     if cv2.contourArea(cnt) >= min_area:
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(result, (x, y), (x + w, y + h), 255, -1)
# Rotundity
y_coords, x_coords = np.where(processed == 255)
for x, y in zip(x_coords, y_coords):
    cv2.circle(result, (x, y), 2, 255, -1)
cv2.imwrite(str(OUTPUT_ROOT / "collector_intersections.png"), result)
