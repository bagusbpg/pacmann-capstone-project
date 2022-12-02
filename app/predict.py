import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_ocr.pipeline import Pipeline
from keras_ocr.tools import read
import pandas as pd
import sys
import tempfile
from tensorflow.keras.models import load_model
from util import response

def preprocessing_predict(imagePath=None, model=None, saveROI=False):
    if not imagePath:
        return None, response(400, 'image file to predict is missing')

    if not model:
        return None, response(400, 'model is missing')

    paramPath = '../train/params.csv'
    if not os.path.exists(paramPath):
        return None, response(500, 'params.csv is missing')
    
    try:
        params = pd.read_csv(paramPath, index_col='channel')
    except:
        return None, response(500, 'failed to read params.csv')
    
    try:
        originalImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    except:
        return None, response(500, 'failed to read image file')
    
    # resize to default size, reshape, and assigning to float64 data type
    image = cv2.resize(originalImage, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.reshape(1, 224, 224, 3).astype(np.float64)

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
    denormalizer = np.array([0.9*originalWidth, 1.1*originalWidth, 0.9*originalHeight, 1.1*originalHeight])
    xmin, xmax, ymin, ymax = np.multiply(coordinate, denormalizer).astype(np.int32)[0]
    if xmax > originalWidth:
        x = originalWidth
    if ymax > originalHeight:
        y = originalHeight
    regionOfInterest = originalImage[ymin:ymax, xmin:xmax]

    if saveROI:
        try:
            newImagePath = imagePath.replace('.jpg', '')
            newImagePath = f'{newImagePath}-roi.jpg'
            cv2.imwrite(newImagePath, regionOfInterest, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        except:
            pass
        
    # recognize text using keras_ocr
    try:
        pipeline = Pipeline()
        finalImages = [read(image) for image in [regionOfInterest]]
        texts = pipeline.recognize(finalImages)
        text = '-'.join([chars[0] for chars in texts[0]])
    except:
        return None, response(500, 'failed to recognize text')

    return text, None

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

    text, error = preprocessing_predict(imagePath, model, True)
    if error:
        sys.exit(error)

    print(text)

    