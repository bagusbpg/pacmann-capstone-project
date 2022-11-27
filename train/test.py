import cv2
# import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import time
from tensorflow.keras.models import load_model
from train import x_train_validate_test_split, augment_dataset, preprocessing, y_train_validate_test_split

def preprocessingTest(dataset):
    if not os.path.exists('./params.csv'):
        sys.exit('params.csv is missing')

    params = pd.read_csv('./params.csv', index_col='channel')

    # zero-mean
    channel0Mean = params.at[0, 'mean']
    channel1Mean = params.at[1, 'mean']
    channel2Mean = params.at[2, 'mean']
    dataset[..., 0] = np.subtract(dataset[..., 0], channel0Mean)
    dataset[..., 1] = np.subtract(dataset[..., 1], channel1Mean)
    dataset[..., 2] = np.subtract(dataset[..., 2], channel2Mean)
    
    # standardization
    channel0Std = params.at[0, 'std']
    channel1Std = params.at[1, 'std']
    channel2Std = params.at[2, 'std']
    dataset[..., 0] = np.divide(dataset[..., 0], channel0Std)
    dataset[..., 1] = np.divide(dataset[..., 1], channel1Std)
    dataset[..., 2] = np.divide(dataset[..., 2], channel2Std)

    return dataset

def rmse(yTest, yPredict):
    return mean_squared_error(yTest, yPredict, squared=False)

def show_image(coordinate, imagePath, denormalize):
    xmin, xmax, ymin, ymax = coordinate
    print(xmin, xmax, ymin, ymax)
    if denormalize:
        xmin, xmax, ymin, ymax = np.multiply(coordinate, 600).astype(np.int32)
    image = cv2.imread(imagePath)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255,0,0), thickness=3)
    if cv2.imwrite('./test.jpg', image):
        print('image saved as test.jpg')

if __name__ == '__main__':
    try:
        model = load_model('./model.h5', compile=False)
        print('model has been loaded successfully')
    except:
        sys.exit('failed to load model')

    _, _, xTestPath = x_train_validate_test_split()
    print('length of pre-augmented test set:', len(xTestPath))

    xTestRaw = augment_dataset(xTestPath, False)
    print('length of post-augmented test set:', len(xTestRaw))

    xTest = preprocessing(xTestRaw, False)
    print('shape of preprocessed test set:', xTest.shape)

    _, _, yTest = y_train_validate_test_split(xTestPath=xTestPath, withAugmentation=False)
    print('shape of test target set:', yTest.shape)

    yPredict = model.predict(xTest)
    print('shape of test predict set:', yPredict.shape)

    rmse = rmse(yTest, yPredict)
    print('root mean square error:', rmse)

    index = [index.replace('../data/images/', '') for index in xTestPath]
    pd.DataFrame(np.multiply(yPredict, 600).astype(np.int32), columns=['xmin', 'xmax', 'ymin', 'ymax'], index=index).to_csv('inference.csv')

    np.random.seed(np.int64(time.time()))
    randomIndex = np.random.randint(len(index))
    show_image(yPredict[randomIndex], xTestPath[randomIndex], True)