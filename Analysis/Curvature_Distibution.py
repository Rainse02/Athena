import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import bbox_artist
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sympy.abc import alpha

from settings import DATA_ROOT, OUTPUT_ROOT

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

plt.rc("font", family="Arial", size=9)


def cartesian_to_polar(points, center):
    dx = points[:, 0] - center[0]
    dy = -points[:, 1] + center[1]  # y of img need to reversal
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    theta = np.unwrap(theta)
    return r, theta


def spiral_func(theta, a, b):
    return a + b * theta


def curvature_archimedean(theta, a, b, x=0, y=0):
    r = a + b * theta
    numerator = r**2 + 2 * b**2  # 2 * b * r
    denominator = (r**2 + b**2) ** 1.5
    return numerator / denominator


def extract_curve_points(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        raise ValueError("No curve found in the binary image")
    points = contours[0].squeeze()
    return points


def fit_spiral_curvature(binary_image, center=None, show_plot=True):
    # Step 1: Extract points on curves
    points = extract_curve_points(binary_image)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Invalid contour shape")

    # Step 2: Center point setting
    if center is None:
        # center = np.mean(points, axis=0)
        center = [613, 603.5]
    # Step 3: Polar & fitting
    r, theta = cartesian_to_polar(points, center)
    params, _ = curve_fit(spiral_func, theta, r)
    a, b = params

    # Step 4: Curvature calculation
    curvature = curvature_archimedean(theta, a, b) / 0.016849769150315919  # mm^-1

    # Step 5: visualization
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], c=curvature, cmap="jet", s=4)
        plt.colorbar(label="Curvature (mm-1)")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.title("Curvature")
        plt.show()

        theta_fit = np.linspace(min(theta), max(theta), 10000)
        r_fit = spiral_func(theta_fit, a, b)
        x_fit = center[0] + r_fit * np.cos(theta_fit)
        y_fit = center[1] + r_fit * np.sin(theta_fit)
        plt.plot(x_fit, y_fit, "r-", label="Fitted Spiral")
        plt.xlim(0, 1226)
        plt.ylim(0, 1207)
        plt.legend()
    # # Save curvature data and fitted standard curve data
    # results1 = {
    #     "Curvature": curvature,
    #     "x": points[:, 0],
    #     "y": points[:, 1],
    # }
    # results2 = {
    #     "x_fit": x_fit,
    #     "y_fit": y_fit
    # }
    # results1 = pd.DataFrame(results1)
    # results2 = pd.DataFrame(results2)
    # with pd.ExcelWriter(OUTPUT_ROOT / "spiral_fit.xlsx",
    #               mode='a', engine='openpyxl') as writer:
    #     results1.to_excel(writer, sheet_name=f"fit", index=True)
    #     results2.to_excel(writer, sheet_name=f"fit_curve", index=True)

    return {"points": points, "curvature": curvature, "a": a, "b": b, "center": center}


# Process curve image to original size
def crop_white(img_origin):
    top, bottom, left, right = 0, 0, 0, 0
    for i in range(img_origin.shape[0]):
        if not np.all(img_origin[i] == 255):
            top = i
            break
    for j in range(img_origin.shape[1]):
        if not np.all(img_origin[:, j] == 255):
            left = j
            break
    for ii in range(img_origin.shape[0])[::-1]:
        if not np.all(img_origin[ii] == 255):
            bottom = ii
            break
    for jj in range(img_origin.shape[1])[::-1]:
        if not np.all(img_origin[:, jj] == 255):
            right = jj
            break
    img_origin = img_origin[top : bottom + 1, left : right + 1]
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img_origin, kernel, iterations=1)
    img_origin = cv2.resize(
        dilated, (1024, 1024), interpolation=cv2.INTER_NEAREST
    )  # [width, height]
    top, bottom, left, right = 92, 91, 101, 101
    border_type = cv2.BORDER_CONSTANT
    # Fill Boundary with black
    value = 0
    padded_image = cv2.copyMakeBorder(
        img_origin, top, bottom, left, right, border_type, value
    )

    return padded_image


def map_features_to_curve3(
    name, curve_points, curvature, feature_values, feature_coords
):
    # Create KDTree for nearest neighbor search
    tree = KDTree(curve_points)

    # Find the nearest curve point for each feature volume
    distances, indices = tree.query(feature_coords, k=1)

    results = {
        "feature_values": feature_values,
        "feature_x": feature_coords[:, 0],
        "feature_y": feature_coords[:, 1],
        "curvature": curvature[indices.flatten()],
        "nearest_curve_x": curve_points[indices.flatten(), 0],
        "nearest_curve_y": curve_points[indices.flatten(), 1],
        "distance_to_curve": distances.flatten(),
    }

    # # Save the feature curvature data
    # results1 = pd.DataFrame(results)
    # with pd.ExcelWriter(OUTPUT_ROOT / "curvature_volume.xlsx",
    #               mode='a', engine='openpyxl') as writer:
    #     results1.to_excel(writer, sheet_name=f"{name}_volumes", index=True)
    return pd.DataFrame(results)


path = str(DATA_ROOT / "images" / "electrode_contour.png")
image_origin = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(image_origin, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = img.astype(np.uint8)
img = crop_white(img)
_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
img = cv2.ximgproc.thinning(img)
# cv2.imwrite(str(OUTPUT_ROOT / "electrode_contour.png"), img)

result = fit_spiral_curvature(img, show_plot=False)
curve_points = result["points"]
curvature = result["curvature"]

feature_name = ["Internal", "Detach", "Surface"]
color = ["Greens", "Reds", "Blues"]
# color = ["seagreen", "Reds", "Blues"]
# color = ["red", "Blues", "Greens"]
color_1d = [[0, 0.95, 0], [0.9, 0.15, 0.15], [0, 0.4, 0.7]]

# Create a main image and an edge image
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.0)
ax_marg = plt.subplot(gs[0])
ax_main = plt.subplot(gs[1])

for i, name in enumerate(feature_name):
    feature = pd.read_excel(DATA_ROOT / "objects" / f"{name}255.xlsx").drop(0)
    volume = np.array(feature.iloc[:, 0])
    feature_point = np.array(feature.iloc[:, 2:4])
    feature_map = map_features_to_curve3(
        name, curve_points, curvature, volume, feature_point
    )
    values = np.asarray(feature_map["feature_values"], dtype=np.float64)
    # values = values * (0.016849769150315919**3) # mm^3
    # feature_map['feature_values'].values
    x = np.asarray(feature_map["curvature"], dtype=np.float64)

    # Upper edge distribution map (one per group)
    # norm_values = (values - values.min()) / (values.max() - values.min() + 1e-6)  # 避免除0
    kde_1d = gaussian_kde(x, weights=values)
    x_grid_1d = np.linspace(x.min(), x.max(), 200)
    y_kde_1d = kde_1d(x_grid_1d)
    y_kde_norm_1d = (y_kde_1d - y_kde_1d.min()) / (y_kde_1d.max() - y_kde_1d.min())
    ax_marg.plot(x_grid_1d, y_kde_norm_1d, color=color_1d[i])
    ax_marg.fill_between(x_grid_1d, y_kde_norm_1d, color=color_1d[i], alpha=0.3)
    ax_marg.set_xlim(x.min(), x.max())
    ax_marg.set_ylim(0, 1.1)
    ax_marg.axis("off")

    # # The one-dimensional KDE diagram above
    # sns.kdeplot(
    #     data=feature_map,
    #     x='curvature',
    #     fill=True,
    #     weights=norm_values,
    #     ax=ax_marg,
    #     color=color_1d[i],
    #     alpha=0.4, # 透明度
    #     common_norm=True,
    #     label=name
    # )

    # # 2D contour KDE map below
    # # Create a two-dimensional kernel density estimator (weighted)
    # values_gs = np.vstack([x, values])
    # kde_2d = gaussian_kde(values_gs, weights=values)
    # # Generate grid points
    # x_grid_2d = np.linspace(x.min(), x.max(), 200)
    # y_grid_2d = np.linspace(values.min(), values.max(), 200)
    # X_2d, Y_2d = np.meshgrid(x_grid_2d, y_grid_2d)
    # positions = np.vstack([X_2d.ravel(), Y_2d.ravel()])
    # # Calculate KDE values and normalize to 0~1
    # Z_2d = kde_2d(positions).reshape(X_2d.shape)
    # Z_norm_2d = (Z_2d - Z_2d.min()) / (Z_2d.max() - Z_2d.min())  # Normalization
    # # Plot
    # ax_main.contourf(X_2d, Y_2d, Z_norm_2d, levels=20, cmap=color[i], alpha=0.5)

    # 2D contour KDE map below
    if name == "Detach":
        sns.kdeplot(
            data=feature_map,
            x="curvature",
            y="feature_values",
            cmap=color[i],
            weights="feature_values",
            ax=ax_main,
            levels=5,
            bw_adjust=0.8,
            thresh=0.65,
            fill=True,
            label=name,
            # alpha=0.5
            gridsize=200,
        )
    elif name == "Surface":
        sns.kdeplot(
            data=feature_map,
            x="curvature",
            y="feature_values",
            cmap=color[i],
            weights="feature_values",
            ax=ax_main,
            levels=5,
            bw_adjust=0.8,
            thresh=0.6,
            fill=True,
            label=name,
            # alpha=0.5
            gridsize=200,
        )
    elif name == "Internal":
        sns.kdeplot(
            data=feature_map,
            x="curvature",
            y="feature_values",  #
            cmap=color[i],
            weights="feature_values",
            ax=ax_main,
            levels=5,
            bw_adjust=0.8,
            thresh=0.85,
            fill=True,
            label=name,
            # alpha=0.5
            gridsize=200,
        )
    else:
        pass
    ax_main.set_xlim(x.min(), x.max())
    ax_main.set_ylim(0, 3500)  # 0, 3500
    ax_main.axis("off")


plt.xlabel("Curvature (1/mm)")
plt.ylabel("Feature Intensity (mm^3)")
# plt.xlim(0.05, 0.65)
# plt.ylim(0, 45000)
# plt.legend()
# plt.title('Curvature-Feature Intensity Density')
for collection in ax_main.collections:
    collection.set_clip_path(ax_main.patch)
plt.savefig(
    OUTPUT_ROOT / "curvature_distribution.svg", format="svg", bbox_inches="tight"
)
plt.show()
