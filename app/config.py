import json
import sys

def loadJSON():
    with open('./config.json', 'r') as f:
        config = json.load(f)
    
    return config

if __name__ == '__main__':
    config = loadJSON()
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