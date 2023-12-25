FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ADD . /neural-lam

WORKDIR /neural-lam

RUN python -m pip install -r requirements.txt
