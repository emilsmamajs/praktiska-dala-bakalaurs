FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "rfdetr[train,loggers]" Pillow "albumentations>=1.4,<2.0"

ENV RFDETR_WORKDIR=/data/RT-DETR/iteracija1

WORKDIR /data/RT-DETR/iteracija1
