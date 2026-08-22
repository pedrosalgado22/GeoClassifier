import cv2

def color_histogram(image_path, bins=16):
    img = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)

    return features