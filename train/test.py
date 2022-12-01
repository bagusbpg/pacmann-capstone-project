import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import time
from tensorflow.keras.models import load_model
from train import x_train_validate_test_split, augment_dataset, y_train_validate_test_split, preprocessing_test

def show_image(coordinate, imagePath, denormalize, randomIndex):
    xmin, xmax, ymin, ymax = coordinate
    print(xmin, xmax, ymin, ymax)
    if denormalize:
        xmin, xmax, ymin, ymax = np.multiply(coordinate, 600).astype(np.int32)
    image = cv2.imread(imagePath)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255,0,0), thickness=3)
    if cv2.imwrite(f'./test-{randomIndex}.jpg', image):
        print(f'image saved as test-{randomIndex}.jpg')

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

    xTest = preprocessing_test(xTestRaw)
    print('shape of preprocessed test set:', xTest.shape)

    _, _, yTest = y_train_validate_test_split(xTestPath=xTestPath, withAugmentation=False)
    print('shape of test target set:', yTest.shape)

    yPredict = model.predict(xTest)
    print('shape of test predict set:', yPredict.shape)

    mse = mean_squared_error(yTest, yPredict)
    print('mean squared error:', mse)

    index = [index.replace('../data/images/', '') for index in xTestPath]
    pd.DataFrame(np.multiply(yPredict, 600).astype(np.int32), columns=['xmin', 'xmax', 'ymin', 'ymax'], index=index).to_csv('inference.csv')

    np.random.seed(np.int64(time.time()))
    randomIndex = np.random.randint(len(index))
    show_image(yPredict[randomIndex], xTestPath[randomIndex], True, randomIndex)