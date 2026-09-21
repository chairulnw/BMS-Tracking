FROM python:3.13-slim

# ffmpeg: clip encoding. git: pip installs from GitHub in requirements.txt.
# libgl1/libglib2.0-0: opencv-python runtime deps on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch explicitly: plain "pip install torch" on Linux still pulls
# nvidia-cudnn/cublas/etc as required deps (several GB) even without a GPU —
# only the dedicated cpu index skips them. Installed first so ultralytics'
# unpinned "torch" dependency is already satisfied with the small build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# Baked in at /opt (outside /app/checkpoints) so the runtime bind-mount of
# ./ai-service/checkpoints doesn't shadow it — that mount is what silently
# turns this folder into an empty directory on some Windows drives.
COPY checkpoints/TransReID /opt/transreid
ENV TRANSREID_REPO_DIR=/opt/transreid
COPY . .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
