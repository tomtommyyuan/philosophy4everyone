# Philosophy for Everyone — container image.
#
# Works on any container host: Fly.io, Render, Railway, Cloud Run, plain
# Docker. This is the recommended deployment for this app, because the whole
# system is a stateful index plus a small server — which is exactly what a
# container is good at and what a serverless function is awkward about.
#
#   docker build -t philo .
#   docker run --rm -p 8000:8000 philo                    # offline demo
#   docker run --rm -p 8000:8000 \
#     -e OPENAI_API_KEY=sk-… -e PHILO_WEB_TOKEN=$(openssl rand -hex 16) \
#     -v philo-data:/data philo                           # real models
#
# The image ships with the texts and an offline index already built, so the
# container answers immediately with no network. Pointing it at a real
# provider re-indexes once on first start into the mounted volume.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PHILO_HOME=/data \
    PHILO_PROVIDER=mock \
    PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY philo ./philo
RUN pip install --no-cache-dir ".[web,fast]"

# Texts and the offline index are baked in at build time: no network needed
# at runtime, and a cold container is answering questions in milliseconds.
RUN philo fetch && philo ingest

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/api/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["docker-entrypoint.sh"]
