FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update -yqq && apt-get install -yqq \
    build-essential python3-pip python3-tqdm

COPY MultiPL-E/evaluation/src /code
COPY benchmark/requirements.txt /code
COPY benchmark/evaluator_overlay.py /code
COPY benchmark/.coveragerc /code
WORKDIR /code
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "evaluator_overlay.py"]

