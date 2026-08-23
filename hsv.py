import cv2
import numpy as np


def vegetation_features(image_path, threshold=0.1):
    img = cv2.imread(str(image_path)).astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    vari = (g - r) / (g + r - b + 1e-6)

    return [vari.mean(), vari.std(), (vari > threshold).mean()]


def color_histogram(image_path, bins=16):
    img = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)

    return features


def vegetation_features(image_path, threshold=0.1):
    img = cv2.imread(str(image_path)).astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    vari = (g - r) / (g + r - b + 1e-6)

    return [vari.mean(), vari.std(), (vari > threshold).mean()]







def sky_features(image_path, v_thresh=200, s_thresh=30, blue_hue_range=(100, 130)):
    img = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    top_third = hsv[:hsv.shape[0] // 3, :, :]
    h, s, v = top_third[:, :, 0], top_third[:, :, 1], top_third[:, :, 2]

    overcast_mask = (v > v_thresh) & (s < s_thresh)
    blue_mask = (h > blue_hue_range[0]) & (h < blue_hue_range[1]) & (v > v_thresh)
    sky_mask = overcast_mask | blue_mask

    sky_fraction = sky_mask.mean()
    sky_blue_ratio = blue_mask.sum() / (sky_mask.sum() + 1e-6)

    return [sky_fraction, sky_blue_ratio]