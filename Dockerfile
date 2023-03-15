FROM python:3.7-bullseye

COPY . .

RUN pip install -U pip \
    && pip install --no-cache-dir gdown -r requirements.txt \
    && cd dss_crf \
    && python setup.py install \
    && cd ..

RUN gdown "1xOwuH9lWizGVnPmEH77_81Sp9EAN054O"

ENTRYPOINT ["python", "infer.py"]
