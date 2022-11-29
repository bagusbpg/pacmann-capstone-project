import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from pytesseract import image_to_string
import sys
from tensorflow.keras.models import load_model
from util import response

def preprocessing_predict(imagePath=None, model=None):
    if not imagePath:
        return response(400, 'image file to predict is missing'), None

    if not model:
        return response(400, 'model is missing'), None

    paramPath = '../train/params.csv'
    if not os.path.exists(paramPath):
        return response(500, 'params.csv is missing'), None
    
    try:
        params = pd.read_csv(paramPath, index_col='channel')
    except:
        return response(500, 'failed to read params.csv'), None
    
    try:
        originalImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    except:
        return response(500, 'failed to read image file'), None
    
    # resize to default size
    image = cv2.resize(originalImage, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.reshape(1, 224, 224, 3)
    
    # zero-mean
    channel0Mean = params.at[0, 'mean']
    channel1Mean = params.at[1, 'mean']
    channel2Mean = params.at[2, 'mean']
    image[..., 0] = np.subtract(image[..., 0], channel0Mean)
    image[..., 1] = np.subtract(image[..., 1], channel1Mean)
    image[..., 2] = np.subtract(image[..., 2], channel2Mean)

    # standardization
    channel0Std = params.at[0, 'std']
    channel1Std = params.at[1, 'std']
    channel2Std = params.at[2, 'std']
    image[..., 0] = np.divide(image[..., 0], channel0Std)
    image[..., 1] = np.divide(image[..., 1], channel1Std)
    image[..., 2] = np.divide(image[..., 2], channel2Std)

    # get region of interest
    coordinate = model.predict(image)
    originalHeight, originalWidth, _ = originalImage.shape
    denormalizer = np.array([originalWidth, originalWidth, originalHeight, originalHeight])
    xmin, xmax, ymin, ymax = np.multiply(coordinate, denormalizer).astype(np.int32)
    regionOfInterest = originalImage[ymin:ymax, xmin:xmax]
    try:
        cv2.imwrite(f'{imagePath}-roi.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    except:
        pass

    # recognize text
    try:
        text = image_to_string(regionOfInterest)
    except:
        text = ''

    if text != '':
        return response(400, 'no license plate found'), None

    return None, text

if __name__ == '__main__':
    try:
        model = load_model('../train/model.h5', compile=False)
        print('model has been loaded successfully')
    except:
        print('failed to load model')

    if len(sys.argv) != 2:
        sys.exit('please input source image file to predict')

    imagePath = sys.argv[1]
    if not os.path.exists(imagePath):
        sys.exit(f'{imagePath} does not exist')
    print('image path:', imagePath)

    response, image, originalShape = preprocessing_predict(imagePath, model)
    if response:
        sys.exit(response)

    regionOfInterest = getRegionOfInterest(model, image, originalShape)

    