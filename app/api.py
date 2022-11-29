from fastapi import FastAPI, UploadFile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from predict import preprocessing_predict
import shutil
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model
import time
from typing import IO
from util import response
import uvicorn

try:
    model = load_model('../train/model.h5', compile=False)
else:
    model = None

app = FastAPI()

@app.get('/')
async def root():
    return response(200, 'service works fine!')

@app.post('/register')
async def register(file: UploadFile | None = None):
    if not file:
        return response(400, 'no image uploaded')
    
    if file.content_type != 'image/jpeg':
        return response(415, 'unallowed image type')
    
    real_file_size = 0
    temp: IO = NamedTemporaryFile(delete=False)
    for chunk in file.file:
        real_file_size += len(chunk)
        if real_file_size > 2_000_000:
            return response(413, 'image size is too large')
        temp.write(chunk)
    temp.close()
    
    filePath = f'./{time.time_ns()}.jpg'
    shutil.move(temp.name, filePath)
    
    response, text = preprocessing_predict(filePath, model)
    if response:
        return response
    
    return response(200, f'liscence plate {text}')

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8080)