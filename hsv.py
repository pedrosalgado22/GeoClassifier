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