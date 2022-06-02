# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /ttn
RUN apt update && apt upgrade -y
RUN apt install gcc libc-dev -y
RUN python3 -m pip install pip --upgrade

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD [ "python3", "example.py"]
