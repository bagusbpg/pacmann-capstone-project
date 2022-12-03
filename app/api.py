from config import loadJSON
import datetime
from db import database_init, check_in, fetch_all, check_out
from fastapi import FastAPI, UploadFile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from predict import preprocessing_predict
from tensorflow.keras.models import load_model
from util import most_similar, prepare_image, response
import uvicorn

try:
    model = load_model('../train/model.h5', compile=False)
    config = loadJSON()
    HOST = config['APP']['HOST']
    PORT = config['APP']['PORT']
    LIMIT = config['APP']['MAXIMUM_IMAGE_UPLOAD_SIZE']
    THRESHOLD = config['APP']['THRESHOLD_OF_SIMILARITY']
except:
    model = None
    config = None

app = FastAPI()
db = database_init(config)

@app.get('/')
async def root():
    return response(200, 'service works fine!')

@app.post('/checkin')
async def check_in_new_car(file: UploadFile | None = None):
    imagePath, error = prepare_image(file, LIMIT)
    if error:
        return error
    
    text, error = preprocessing_predict(imagePath, model)
    if error:
        os.remove(imagePath)
        return error

    os.remove(imagePath)
    checkedInCars, error = fetch_all(db, False)
    if error:
        return error

    if len(checkedInCars) != 0:
        existingId, _ = most_similar(checkedInCars, text, THRESHOLD)
        if existingId:
            return response(400, f'liscence plate{text} already checked in with id {existingId}')

    id, error = check_in(db, text)
    if error:
        return error
            
    return response(200, f'licence plate {text} checked in with id {id}')

@app.post('/checkout')
async def check_out_existing_car(file: UploadFile | None = None):
    imagePath, error = prepare_image(file, LIMIT)
    if error:
        return error

    text, error = preprocessing_predict(imagePath, model)
    if error:
        os.remove(imagePath)
        return error

    os.remove(imagePath)
    checkedInCars, error = fetch_all(db, False)
    if error:
        return error

    if len(checkedInCars) == 0:
        return response(400, 'car has not been checked in before')

    existingId, error = most_similar(checkedInCars, text, THRESHOLD)
    if error:
        return error
    
    checkOutTime, error = check_out(db, existingId, text)
    dateTime = datetime.datetime.fromtimestamp(checkOutTime).strftime('%c')
    if error:
        return error

    return response(200, f'licence plate {text} checked out at {dateTime}')

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)