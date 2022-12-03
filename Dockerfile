FROM python:3.10.8-slim-buster
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python ./app/api.py