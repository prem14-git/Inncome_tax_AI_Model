# Stage 1: build environment
FROM python:3.10-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefer-binary torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache/pip /root/.cache/huggingface

# Stage 2: runtime image - much smaller
FROM python:3.10-slim as runtime
WORKDIR /app
# Copy only site-packages and scripts
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source
COPY . .

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "app:app", "--bind", "0.0.0.0:8000"]