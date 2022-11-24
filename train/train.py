import cv2
from glob import glob
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
# from tensorflow.image import rot90
# from tensorflow.keras.utils import load_img, img_to_array

def train_validate_test_split(imageDir):
    allImagePaths = glob(f'{imageDir}/*.jpg')
    xTrainPath, xValidateTestPath = train_test_split(allImagePaths, test_size=0.3, random_state=42)
    xValidatePath, xTestPath = train_test_split(xValidateTestPath, test_size=0.5, random_state=42)

    return xTrainPath, xValidatePath, xTestPath

def augment_dataset(path):
    dataset = np.ndarray(shape=(4*len(path), 224, 224, 3), dtype=np.float32)
    for i, imagePath in enumerate(path):
        # default image
        originalImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        image = cv2.resize(originalImage, (224,224), interpolation = cv2.INTER_AREA)
        dataset[4*i] = image
        # image rotated 90-deg clockwise
        imageRot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        dataset[4*i+1] = imageRot90
        # image rotated 180-deg clockwise
        imageRot180 = cv2.rotate(imageRot90, cv2.ROTATE_90_CLOCKWISE)
        dataset[4*i+2] = imageRot180
        # image rotated 270-deg clockwise
        imageRot270 = cv2.rotate(imageRot180, cv2.ROTATE_90_CLOCKWISE)
        dataset[4*i+3] = imageRot270

    return dataset

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('please input source directory')
    
    imageDir = sys.argv[1]
    if not os.path.exists(imageDir):
        sys.exit(f'directory {imageDir} does not exist')
    print('image directory:', imageDir)

    xTrainPath, xValidatePath, xTestPath = train_validate_test_split(imageDir)
    print('length of pre-augmented training set:', len(xTrainPath))
    print('length of pre-augmented validation set:', len(xValidatePath))
    print('length of pre-augmented test set:', len(xTestPath))

    xTrainRaw = augment_dataset(xTrainPath)
    xValidateRaw = augment_dataset(xValidatePath)
    xTestRaw = augment_dataset(xTestPath)
    print('length of post-augmented training set:', len(xTrainRaw))
    print('length of post-augmented validation set:', len(xValidateRaw))
    print('length of post-augmented test set:', len(xTestRaw))