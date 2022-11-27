import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    try:
        model = load_model('../train/model.h5', compile=False)
        print('model has been loaded successfully')
    except:
        print('failed to load model')
