# CPU image by default. For a GPU host build with:
#   docker build --build-arg BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime --build-arg INSTALL_TORCH=0 .
ARG BASE=python:3.11-slim
FROM ${BASE}
WORKDIR /app
ARG INSTALL_TORCH=1
RUN if [ "$INSTALL_TORCH" = "1" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; fi
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
COPY serve/ serve/
COPY ui/ ui/
ENV HF_HOME=/models
EXPOSE 8000 8501
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
