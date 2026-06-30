# GPU base image — needs an nvidia-container-toolkit host (RunPod, Vast.ai,
# and GCP's GPU-enabled images all provide this already).
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the inference engine itself
RUN git clone https://github.com/openvpi/DiffSingerMiniEngine.git .
RUN pip3 install --no-cache-dir -r requirements.txt

# Then the API wrapper layer
COPY requirements.txt /app/wrapper-requirements.txt
RUN pip3 install --no-cache-dir -r wrapper-requirements.txt

COPY server.py /app/server.py

# Model checkpoints are multi-GB and change independently of the code —
# bake them into the image only if you want fully self-contained deploys.
# Otherwise mount them at runtime, e.g.:
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/assets:/app/assets \
#     harmony-diffsinger
COPY assets /app/assets

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
