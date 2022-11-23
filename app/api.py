from fastapi import FastAPI, UploadFile
import shutil
from tempfile import NamedTemporaryFile
import time
from typing import IO
import uvicorn

app = FastAPI()

@app.get('/')
async def root():
    return {
        'code': 200,
        'message': 'service works fine!'
    }

@app.post('/upload')
async def predict(file: UploadFile | None = None):
    if not file:
        return {
            'code': 400,
            'message': 'no image uploaded'
        }
    if file.content_type != 'image/jpeg':
        return {
            'code': 415,
            'message': 'unallowed image type'
        }
    real_file_size = 0
    temp: IO = NamedTemporaryFile(delete=False)
    for chunk in file.file:
        real_file_size += len(chunk)
        if real_file_size > 2_000_000:
            return {
                'code': 413,
                'message': 'image size is too large'
            }
        temp.write(chunk)
    temp.close()
    shutil.move(temp.name, f'./{time.time_ns()}.jpg')
    
    return {
        'code': 200,
        'messsage': 'image received'
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8080)