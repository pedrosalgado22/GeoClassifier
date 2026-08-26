import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

def color_histogram(image_path, bins=16):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = []
    for channel in range(3):
        max_val = 180 if channel == 0 else 256
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, max_val])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)

    return features


def vegetation_features(image_path, threshold=0.1):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = img.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    vari = (g - r) / (g + r - b + 1e-6)

    return [vari.mean(), vari.std(), (vari > threshold).mean()]




def sky_features(image_path, v_thresh=200, s_thresh=30, blue_hue_range=(100, 130)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    top_third = hsv[:hsv.shape[0] // 3, :, :]
    h, s, v = top_third[:, :, 0], top_third[:, :, 1], top_third[:, :, 2]

    overcast_mask = (v > v_thresh) & (s < s_thresh)
    blue_mask = (h > blue_hue_range[0]) & (h < blue_hue_range[1]) & (v > v_thresh)
    sky_mask = overcast_mask | blue_mask

    sky_fraction = sky_mask.mean()
    sky_blue_ratio = blue_mask.sum() / (sky_mask.sum() + 1e-6)

    return [sky_fraction, sky_blue_ratio]




from skimage.feature import hog
import pandas as pd


# grayscale (edges/shape, not color) + resize to 128x128 (keep feature count manageable)
# pixels_per_cell=(32,32) on 128x128 -> 4x4 grid of cells, cells_per_block=(1,1) -> 144 features
def extract_hog(df, resize_dim=(128, 128), pixels_per_cell=(32, 32), cells_per_block=(1, 1), orientations=9):
    results = []
    for path in df['path']:
        img = cv2.imread(str(path))
        img = cv2.resize(img, resize_dim)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        results.append(features)

    hog_cols = pd.DataFrame(results, columns=[f'hog_{i}' for i in range(len(results[0]))])
    df = pd.concat([df.reset_index(drop=True), hog_cols], axis=1)
    return df 



def extract_lbp(df, P=8, R=1, grid=(2, 2), resize_dim=(128, 128)):
    results = []
    n_bins = P + 2

    for path in df['path']:
        img = cv2.imread(str(path))
        img = cv2.resize(img, resize_dim)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lbp = local_binary_pattern(gray, P, R, method='uniform')

        h, w = lbp.shape
        gh, gw = h // grid[0], w // grid[1]

        cell_hists = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                cell = lbp[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))
                cell_hists.extend(hist / hist.sum())

        results.append(cell_hists)

    lbp_cols = pd.DataFrame(results, columns=[f'lbp_{i}' for i in range(len(results[0]))])
    df = pd.concat([df.reset_index(drop=True), lbp_cols], axis=1)
    return df



def extract_ground(df):
    results = []
    for path in df['path']:
        img = cv2.imread(str(path))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        bottom = hsv[-hsv.shape[0]//3:, :, :]
        h, s, v = bottom[:,:,0], bottom[:,:,1], bottom[:,:,2]
        grey_mask = (s < 30) & (v > 50) & (v < 200)

        results.append([h.mean(), s.mean(), v.mean(), grey_mask.mean()])

    cols = pd.DataFrame(results, columns=['ground_hue_mean', 'ground_sat_mean', 'ground_val_mean', 'ground_grey_frac'])
    df = pd.concat([df.reset_index(drop=True), cols], axis=1)
    return df



def extract_contrast(df):
    results = []
    for path in df['path']:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        results.append([gray.mean(), gray.std()])

    cols = pd.DataFrame(results, columns=['brightness_mean', 'contrast_std'])
    df = pd.concat([df.reset_index(drop=True), cols], axis=1)
    return df



def extract_edges(df):
    results = []
    for path in df['path']:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        top, bottom = edges[:edges.shape[0]//2, :], edges[edges.shape[0]//2:, :]
        results.append([(edges > 0).mean(), (top > 0).mean(), (bottom > 0).mean()])

    cols = pd.DataFrame(results, columns=['edge_density_total', 'edge_density_top', 'edge_density_bottom'])
    df = pd.concat([df.reset_index(drop=True), cols], axis=1)
    return df