# pull official base image
FROM python:3.9.5-slim-buster

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get -y install libsndfile-dev
RUN pip install --upgrade redis
COPY ./dist/lookup-0.1.0-py3-none-any.whl .
RUN pip install lookup-0.1.0-py3-none-any.whl

# copy project
COPY . .
