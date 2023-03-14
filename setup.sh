#!/bin/bash


rm -rf venv
$1 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt
cd dss_crf
python setup.py install
cd ..

deactivate
