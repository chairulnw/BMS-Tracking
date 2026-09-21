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
# CPU-only torch+torchvision explicitly, both from the same index: plain
# "pip install torch" on Linux still pulls nvidia-cudnn/cublas/etc as required
# deps (several GB) even without a GPU — only the dedicated cpu index skips
# them. torchvision must come from here too, or pip resolves it separately
# from default PyPI and the two builds mismatch ("operator torchvision::nms
# does not exist"). Installed first so ultralytics' unpinned deps are already
# satisfied with the small, matched build.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# Baked in at /opt (outside /app/checkpoints) so the runtime bind-mount of
# ./ai-service/checkpoints doesn't shadow it — that mount is what silently
# turns this folder into an empty directory on some Windows drives.
COPY checkpoints/TransReID /opt/transreid
ENV TRANSREID_REPO_DIR=/opt/transreid
COPY . .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
