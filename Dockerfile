FROM python:3.10.8-slim-buster
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y
RUN python -m pip install --upgrade pip
COPY . .
RUN pip install -r requirements.txt
CMD python ./app/api.py