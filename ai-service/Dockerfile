FROM python:3.13-slim

# ffmpeg: clip encoding. git: pip installs from GitHub in requirements.txt.
# libgl1/libglib2.0-0: opencv-python runtime deps on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# setuptools<81: newer setuptools dropped pkg_resources, which lap's legacy
# setup.py still imports at build time (fails with "No module named pkg_resources").
# scipy first: torchreid's setup.py imports it implicitly (see CLAUDE.md).
RUN pip install --no-cache-dir "setuptools<81" scipy && pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
