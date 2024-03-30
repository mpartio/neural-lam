FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install git vim texlive cm-super && apt -y clean all

WORKDIR /

RUN git clone -b meps_analysis https://github.com/mpartio/neural-lam.git

WORKDIR /neural-lam

RUN python -m pip install -r requirements.txt
