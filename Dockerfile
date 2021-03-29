# Use python base image
#FROM continuumio/anaconda3
FROM python:3.7.10

# Copy over the requirements for the docker container
COPY requirements.txt /

# Install the requirements
RUN pip install -r /requirements.txt

# Copy over the scripts and the model_metrics
COPY ./model_metrics/ /model_metrics
COPY ./scripts/ /scripts


# Set the current working directory to scripts
WORKDIR ./scripts
