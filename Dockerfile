FROM python:3.7.6
MAINTAINER Miguel Suau <miguel.suau@gmail.com>

ENV SUMO_HOME="$PWD/sumo"
ENV PYTHONPATH="${PYTHONPATH}:${SUMO_HOME}/tools"
ENV PATH="${PATH}:${SUMO_HOME}"
ENV PIPENV_VENV_IN_PROJECT=1


# Sumo install
RUN apt-get update -y \
    && apt-get install -y cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig \
    && git clone --recursive https://github.com/eclipse/sumo \
    && mkdir sumo/build/cmake-build && cd sumo/build/cmake-build \
    && cmake ../.. \
    && make -j$(nproc)

WORKDIR ./

RUN git clone https://github.com/INFLUENCEorg/flow.git

RUN pip install pipenv

COPY ./ ./

RUN pipenv install --python=3.7.6 --deploy --ignore-pipfile

RUN chmod +x /runners/experiment.py
