FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY . /app/
RUN pip install -e .[deepspeed]

VOLUME [ "/app/data", "/app/output" ]
