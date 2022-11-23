import cv2
import numpy as np
import os
import pandas as pd
import sys
from tensorflow.image import rot90
from tensorflow.keras.utils import load_img, img_to_array, save_img

def preprocessing(imageDir):
    for imagePath in os.listdir(imageDir)[:2]:
        if imagePath.endswith('jpg'):
            print(imagePath)
            # default image
            image = load_img(f'{imageDir}/{imagePath}', target_size=(224, 224))
            imageArray = img_to_array(image)
            normalizedImageArray = np.divide(imageArray, 255)
            
            # image rotated 90-deg clockwise
            imageRot90 = rot90(image, k=3)
            save_img('test.jpg', imageRot90)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('please input source directory')
    
    imageDir = sys.argv[1]
    if not os.path.exists(imageDir):
        sys.exit(f'directory {imageDir} does not exist')
    print('image directory:', imageDir)

    preprocessing(imageDir)