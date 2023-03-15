FROM python:3.7-bullseye

COPY . .

RUN pip install -U pip \
    && pip install -r requirements.txt --no-cache-dir \
    && cd dss_crf \
    && python setup.py install \
    && cd ..

ENTRYPOINT ["bash", "eval.sh", "--path_to_pretrained_model=200.pth", "--input_dir=input"]
