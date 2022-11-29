import cv2
from glob import glob
import numpy as np
np.random.seed(42)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam

def x_train_validate_test_split():
    allImagePaths = glob('../data/images/*.jpg')
    xTrainPath, xValidateTestPath = train_test_split(allImagePaths, test_size=0.3, random_state=42)
    xValidatePath, xTestPath = train_test_split(xValidateTestPath, test_size=0.5, random_state=42)

    return xTrainPath, xValidatePath, xTestPath

def augment_dataset(path, augment):
    dataset = np.ndarray(shape=(len(path), 224, 224, 3), dtype=np.float64)
    if augment:
        dataset = np.ndarray(shape=(4*len(path), 224, 224, 3), dtype=np.float64)
    
    for index, imagePath in enumerate(path):
        originalImage = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        image = cv2.resize(originalImage, (224, 224), interpolation=cv2.INTER_AREA)
        if augment:
            # default image
            dataset[4*index] = image
            # image rotated 90-deg clockwise
            imageRot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            dataset[4*index+1] = imageRot90
            # image rotated 180-deg clockwise
            imageRot180 = cv2.rotate(imageRot90, cv2.ROTATE_90_CLOCKWISE)
            dataset[4*index+2] = imageRot180
            # image rotated 270-deg clockwise
            imageRot270 = cv2.rotate(imageRot180, cv2.ROTATE_90_CLOCKWISE)
            dataset[4*index+3] = imageRot270
        else:
            dataset[index] = image

    return dataset

def y_train_validate_test_split(xTrainPath=None, xValidatePath=None, xTestPath=None, withAugmentation=False):
    target = pd.read_csv('../data/target.csv', index_col='id')

    yTrain, yValidate, yTest = None, None, None

    if xTrainPath:
        yTrainIndex = [path.replace('../data/images/', '') for path in xTrainPath]
        yTrainDefault = np.divide(target.loc[yTrainIndex,:].values, 600)
        yTrain = np.ndarray(shape=(len(yTrainDefault), 4), dtype=np.float64)
        if withAugmentation:
            yTrain = np.ndarray(shape=(4*len(yTrainDefault), 4), dtype=np.float64)
        for index, value in enumerate(yTrainDefault):
            xmin, xmax, ymin, ymax = value
            if withAugmentation:
                yTrain[4*index] = [xmin, xmax, ymin, ymax]              # default coordinate
                yTrain[4*index+1] = [1-ymax, 1-ymin, xmin, xmax]        # coordinate rotated 90-deg clockwise
                yTrain[4*index+2] = [1-xmax, 1-xmin, 1-ymax, 1-ymin]    # coordinate rotated 180-deg clockwise
                yTrain[4*index+3] = [ymin, ymax, 1-xmax, 1-xmin]        # coordinate rotated 270-deg clockwise
            else:
                yTrain[index] = [xmin, xmax, ymin, ymax]

    if xValidatePath:
        yValidateIndex = [path.replace('../data/images/', '') for path in xValidatePath]
        yValidateDefault = np.divide(target.loc[yValidateIndex,:].values, 600)
        yValidate = np.ndarray(shape=(len(yValidateDefault), 4), dtype=np.float64)
        if withAugmentation:
            yValidate = np.ndarray(shape=(4*len(yValidateDefault), 4), dtype=np.float64)
        for index, value in enumerate(yValidateDefault):
            xmin, xmax, ymin, ymax = value
            if withAugmentation:
                yValidate[4*index] = [xmin, xmax, ymin, ymax]           # default coordinate
                yValidate[4*index+1] = [1-ymax, 1-ymin, xmin, xmax]     # coordinate rotated 90-deg clockwise
                yValidate[4*index+2] = [1-xmax, 1-xmin, 1-ymax, 1-ymin] # coordinate rotated 180-deg clockwise
                yValidate[4*index+3] = [ymin, ymax, 1-xmax, 1-xmin]     # coordinate rotated 270-deg clockwise
            else:
                yValidate[index] = [xmin, xmax, ymin, ymax]

    if xTestPath:
        yTestIndex = [path.replace('../data/images/', '') for path in xTestPath]
        yTestDefault = np.divide(target.loc[yTestIndex,:].values, 600)
        yTest = np.ndarray(shape=(len(yTestDefault), 4), dtype=np.float64)
        if withAugmentation:
            yTest = np.ndarray(shape=(4*len(yTestDefault), 4), dtype=np.float64)
        for index, value in enumerate(yTestDefault):
            xmin, xmax, ymin, ymax = value
            if withAugmentation:
                yTest[4*index] = [xmin, xmax, ymin, ymax]               # default coordinate
                yTest[4*index+1] = [1-ymax, 1-ymin, xmin, xmax]         # coordinate rotated 90-deg clockwise
                yTest[4*index+2] = [1-xmax, 1-xmin, 1-ymax, 1-ymin]     # coordinate rotated 180-deg clockwise
                yTest[4*index+3] = [ymin, ymax, 1-xmax, 1-xmin]         # coordinate rotated 270-deg clockwise
            else:
                yTest[index] = [xmin, xmax, ymin, ymax]
    
    return yTrain, yValidate, yTest

def preprocessing(dataset, dumped=False):
    # zero-mean
    channel0Mean = np.mean(dataset[..., 0])
    channel1Mean = np.mean(dataset[..., 1])
    channel2Mean = np.mean(dataset[..., 2])
    dataset[..., 0] = np.subtract(dataset[..., 0], channel0Mean)
    dataset[..., 1] = np.subtract(dataset[..., 1], channel1Mean)
    dataset[..., 2] = np.subtract(dataset[..., 2], channel2Mean)
    
    # standardization
    channel0Std = np.std(dataset[..., 0])
    channel1Std = np.std(dataset[..., 1])
    channel2Std = np.std(dataset[..., 2])
    dataset[..., 0] = np.divide(dataset[..., 0], channel0Std)
    dataset[..., 1] = np.divide(dataset[..., 1], channel1Std)
    dataset[..., 2] = np.divide(dataset[..., 2], channel2Std)

    # dump train parameters
    if dumped:
        params = {
            'channel': [0, 1, 2],
            'mean': [channel0Mean, channel1Mean, channel2Mean],
            'std': [channel0Std, channel1Std, channel2Std]
        }
        pd.DataFrame(params).set_index('channel').to_csv('params.csv')

    return dataset

def preprocessing_test(dataset):
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

def model_definition():
    inception_resnet = InceptionResNetV2(
        weights='imagenet', 
        include_top=False, 
        input_tensor=Input(shape=(224,224,3))
    )
    headmodel = inception_resnet.output
    headmodel = Flatten()(headmodel)
    headmodel = Dense(500,activation='relu')(headmodel)
    headmodel = Dense(250,activation='relu')(headmodel)
    headmodel = Dense(4,activation='sigmoid')(headmodel)
    model = Model(inputs=inception_resnet.input, outputs=headmodel)
    model.compile(loss='mse', optimizer=Adam(learning_rate=5e-5))

    return model

def fit(model, xTrain, yTrain, xValidate, yValidate):
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        x=xTrain,
        y=yTrain,
        batch_size=10,
        epochs=30,
        validation_data=(xValidate,yValidate),
        callbacks=[earlyStopping]
    )

    save_model(model, './model.h5', include_optimizer=False, save_format='h5')
    pd.DataFrame(history.history).to_csv('train_log.csv')

if __name__ == '__main__':
    xTrainPath, xValidatePath, _ = x_train_validate_test_split()
    print('length of pre-augmented train set:', len(xTrainPath))
    print('length of pre-augmented validation set:', len(xValidatePath))

    xTrainRaw = augment_dataset(xTrainPath, True)
    xValidateRaw = augment_dataset(xValidatePath, True)
    print('length of post-augmented train set:', len(xTrainRaw))
    print('length of post-augmented validation set:', len(xValidateRaw))

    xTrain = preprocessing(xTrainRaw, True)
    xValidate = preprocessing_test(xValidateRaw)
    print('shape of preprocessed train set:', xTrain.shape)
    print('shape of preprocessed validation set:', xValidate.shape)

    yTrain, yValidate, _ = y_train_validate_test_split(xTrainPath, xValidatePath, withAugmentation=True)
    print('shape of train target set:', yTrain.shape)
    print('shape of validation target set:', yValidate.shape)

    model = model_definition()
    fit(model, xTrain, yTrain, xValidate, yValidate)
