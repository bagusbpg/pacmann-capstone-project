FROM python:3.9.15-slim-buster
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python ./app/api.py