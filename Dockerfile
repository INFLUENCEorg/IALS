FROM python:3.7
MAINTAINER Miguel Suau <miguel.suau@gmail.com>

WORKDIR ./

RUN git clone https://github.com/INFLUENCEorg/flow

COPY requirements.txt /requirements.txt

RUN pip3 install -r ./requirements.txt
RUN pip3 install ./flow

COPY ./ ./
