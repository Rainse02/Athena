import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

from imagecodecs import none_decode
from scipy.stats import gaussian_kde
from numpy.array_api import astype
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, Colormap
from PIL import Image
from scipy.ndimage import binary_dilation
from matplotlib.cm import ScalarMappable
import io

def enhance_density_contrast(density_map, gamma=0.5):
    """Apply gamma correction to enhance contrast (0 < gamma < 1: brighter hotspots)"""
    return np.power(density_map, gamma)

def binary_to_density(binary_img, kernel_size=121, sigma=100):
    """
    Creating density fields with Gaussian blur
    Parameters:
        Binary_img: binary image (0 or 255)
        Kernel_size: Gaussian kernel size (odd)
        Sigma: standard deviation
    Return:
        Normalized density map (0-1)
    """
    # dilate the small area
    dilated = binary_dilation(binary_img, structure=np.ones((16, 16)))

    # Convert to floating point and normalize
    binary_float = dilated.astype(np.float32) / 255.0

    # Apply the Gaussian blur
    density = cv2.GaussianBlur(binary_float, (kernel_size, kernel_size), sigma)
    # density = binary_to_density(density)

    # Normalized to 0-1
    return (density - density.min()) / (density.max() - density.min())


def distance_transform_density(binary_img, max_dist=100):
    """
    Density field based on distance transformation
    Parameters:
        Binary_img: binary image
        Max_dist: maximum influence distance (pixels)
    Return:
        Range field density map
    """
    # Make sure the input is a binary image
    _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)

    # Calculate distance transformation
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Normalize and reverse (maximize foreground density)
    density = np.clip(dist / max_dist, 0, 1)
    return density


def sliding_window_density(binary_img, window_size=64):
    """
    Calculate local density through sliding window
    Parameters:
        Window_size: statistics window side length (pixels)
    Return:
        Local density map
    """
    # Convert to 0-1 binary graph
    binary = (binary_img > 127).astype(np.float32)

    # Define normalization window
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    # Convolution calculation of local density
    return cv2.filter2D(binary, -1, kernel)

def kde_count(binary_img):
    """
    Calculate local density through kde
    Parameters:
        none
    Return:
        Local density map
    """
    # get location
    y, x = np.where(binary_img > 0)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.05)

    # thermogram
    grid_x, grid_y = np.meshgrid(np.arange(binary_img.shape[1]),
                                 np.arange(binary_img.shape[0]))
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = kde(grid_coords).reshape(binary_img.shape)

    # normalize
    density = (density - density.min()) / (density.max() - density.min())
    return density


def morphological_density(binary_img, iterations=5):
    """
    Generating density field by morphological operation
    Parameters:
        Iterations: expansion/corrosion times
    Return:
        Morphological density map
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(binary_img, kernel, iterations=iterations)
    eroded = cv2.erode(binary_img, kernel, iterations=iterations)

    # Calculate gradient field
    return (dilated.astype(float) - eroded.astype(float)) / 255.0


path = r"E:\Work\Electrode_fault_identification\Count\Detach\Color_resize\Detach out_resize_color.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = img
gray = np.where(gray != 0, 255, gray)
# GRADIENT_IMAGE = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# Convolution operation
kernel = np.ones((33, 33), dtype=np.float32)

conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
# conv_abs = cv2.normalize(np.abs(conv), None, 0, 255, cv2.NORM_MINMAX)
conv_normalization = ((conv / conv.max())*255).astype(np.uint8)

GRADIENT_IMAGE = cv2.applyColorMap(conv_normalization, cv2.COLORMAP_JET)

result1 = binary_to_density(gray)
# result2 = distance_transform_density(gray)
# result3 = sliding_window_density(gray)
# result4 = morphological_density(gray)
# result5 = kde_count(gray) # costs very very long time

# plt.subplot(4, 1, 1)
# plt.imshow(result1, cmap='jet')
# plt.colorbar(label='Density')
# plt.subplot(4, 1, 2)
# plt.imshow(result2, cmap='viridis')# jet/viridis/plasma
# plt.colorbar(label='Density')
# plt.subplot(4, 1, 3)
# plt.imshow(result3, cmap='plasma')# jet/viridis/plasma
# plt.colorbar(label='Density')
# plt.subplot(4, 1, 4)
# plt.imshow(result4, cmap='jet')# jet/viridis/plasma
# plt.colorbar(label='Density')
# plt.show()

def array_to_image(data, cmap='viridis', vmin=None, vmax=0.64):
    """Convert array to PIL image (original size)"""
    norm = Normalize(vmin=data.min() if vmin is None else vmin,
                    vmax=data.max() if vmax is None else vmax)
    cmap = plt.get_cmap(cmap)
    img_data = cmap(norm(data), bytes=True)  # Convert to RGBA (0-255)
    return Image.fromarray(img_data[:, :, :3])  # Remove alpha channel

# Apply
img = array_to_image(result1, cmap='jet')#jet
img.save('E:/DESKTOP/sur out.png')