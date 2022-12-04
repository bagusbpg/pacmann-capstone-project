import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from tensorflow.keras.models import load_model

def load_my_model(path='../train/model.h5'):
    if not os.path.exists(path):
        path = '../train/model.h5'

    try:
        model = load_model(path, compile=False)

        return model
    except:
        return None


def load_JSON(path='../config.json'):
    if not os.path.exists(path):
        path = '../config.json'

    try:
        with open(path, 'r') as f:
            config = json.load(f)
        
        return config
    except:
        return None

if __name__ == '__main__':
    config = load_JSON()
    if len(sys.argv) == 1:
        print(config)
    elif len(sys.argv) == 2:
        try:
            print(f'config[{sys.argv[1]}]: {config[sys.argv[1]]}')
        except KeyError:
            print(f'key {sys.argv[1]} does not exist')
    else:
        try:
            configFirstKey = config[sys.argv[1]]
            try:
                print(f'config[{sys.argv[1]}][{sys.argv[2]}]: {configFirstKey[sys.argv[2]]}')
            except KeyError:
                print(f'key {sys.argv[2]} does not exist')
        except KeyError:
            print(f'key {sys.argv[1]} does not exist')