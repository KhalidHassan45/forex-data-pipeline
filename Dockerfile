FROM python:3.12-slim

# Node 20 for dukascopy-node + curl/flock for the wrapper
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates util-linux gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g dukascopy-node@1 \
 && apt-get purge -y gnupg && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /root/.npm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fetch_forex.py run.sh schema.sql ./
COPY backtest/ /app/backtest/
COPY importer/ /app/importer/
COPY dashboard/ /app/dashboard/
COPY research/ /app/research/
RUN chmod +x run.sh /app/backtest/run_lab.sh /app/dashboard/build.sh /app/research/research.sh

ENV DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    TZ=UTC
VOLUME ["/data"]

# Idle worker: Coolify Scheduled Tasks exec into this container to run jobs.
CMD ["sleep", "infinity"]
