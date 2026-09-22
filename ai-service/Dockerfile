FROM python:3.13-slim

# unbuffered stdout: without this, print() (main.py's timestamped log lines)
# sits in a block buffer and shows up in `docker logs` minutes late or not
# until the process exits — stdout isn't a TTY inside a container.
ENV PYTHONUNBUFFERED=1

# ffmpeg: clip encoding. git: pip installs from GitHub in requirements.txt.
# libgl1/libglib2.0-0: opencv-python runtime deps on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CUDA-enabled torch+torchvision explicitly, both from the same index: plain
# "pip install torch" resolves torchvision separately from default PyPI,
# and the two builds can mismatch ("operator torchvision::nms does not
# exist") unless both come from here. Installed first so ultralytics'
# unpinned deps are already satisfied with the matched build.
#
# ponytail: switch to https://download.pytorch.org/whl/cpu instead if
# deploying to a machine with no NVIDIA GPU — TransReID (ViT-B) on CPU is
# heavy enough to make the whole service unresponsive under real load
# (observed: 502s / connection-refused from backend while ai-service was
# "Up" but pegged doing CPU inference).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
