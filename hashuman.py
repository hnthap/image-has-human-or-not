import argparse
import joblib
import os
from pathlib import PurePath
import pickle
import requests
import warnings

import cv2 as cv
import numpy as np
from skimage import feature
from sklearn.base import InconsistentVersionWarning
from sklearn.svm import LinearSVC


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nothing here")

    parser.add_argument('url', type=str, help='URL to the image')

    return parser.parse_args()


def crop_square(img_arg: np.ndarray) -> np.ndarray:
    height, width = img_arg.shape
    side = min(height, width)
    return img_arg[0:side, 0:side].copy()


def resize_image(img_arg: np.ndarray) -> np.ndarray:
    return cv.resize(img_arg, (128, 128))


def compute_hog(img_arg: np.ndarray) -> np.ndarray:
    return feature.hog(img_arg, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=False, transform_sqrt=True
    )


def main():
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

    args = parse_arguments()
    url = args.url
    
    filename = PurePath(url).name

    res = requests.get(url)
    if res.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(res.content)

    img = cv.imread(filename, 0)
    os.remove(filename)

    cropped_img = crop_square(img)

    resized_cropped_img = resize_image(cropped_img)
    hog = compute_hog(resized_cropped_img)

    model_1: LinearSVC = joblib.load('svm_model_hog.joblib')

    pred_1 = model_1.predict([hog])[0]

    print(pred_1)


if __name__ == '__main__':
    main()
