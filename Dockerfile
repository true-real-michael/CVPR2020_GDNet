FROM ubuntu:22.04
FROM python:3.7-bullseye

COPY . .

RUN pip install -U pip \
    && pip install -r requirements.txt \
    && cd dss_crf \
    && python setup.py install \
    && cd ..

ENTRYPOINT ["bash", "eval.sh"]
