FROM python:3.7
MAINTAINER Miguel Suau <miguel.suau@gmail.com>

WORKDIR ./

RUN git clone https://github.com/INFLUENCEorg/flow.git

COPY requirements.txt /requirements.txt

RUN pip install -r ./requirements.txt

RUN pip install ./flow

COPY ./ ./

RUN chmod +x /runners/experiment.py
