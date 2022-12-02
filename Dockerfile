FROM python:3.9.15-slim-buster
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD python ./app/api.py