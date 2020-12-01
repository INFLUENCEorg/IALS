FROM python:3.7
MAINTAINER Miguel Suau <miguel.suau@gmail.com>

RUN git clone https://github.com/INFLUENCEorg/flow

RUN pip install -r /requirements.txt
