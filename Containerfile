FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt -y install git vim texlive cm-super

WORKDIR /

RUN git clone https://github.com/mpartio/neural-lam.git

WORKDIR /neural-lam

RUN python -m pip install -r requirements.txt
