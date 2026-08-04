#!/bin/sh
# Ensure the index matches the provider we are about to serve with.
#
# The image bakes in an offline index. If the container is started against a
# real provider, those vectors are from a different embedding model and the
# store will (correctly) refuse to load them — so we rebuild once, into the
# data volume, before the server starts. Without this the first request
# would fail with a model-mismatch error that looks like a bug.
set -e

: "${PHILO_HOME:=/data}"
: "${PORT:=8000}"

if [ ! -d "$PHILO_HOME/library" ] || [ -z "$(ls -A "$PHILO_HOME/library" 2>/dev/null)" ]; then
  echo "philo: fetching texts into $PHILO_HOME/library"
  philo fetch >/dev/null
fi

if ! philo doctor 2>/dev/null | grep -q "✓    index"; then
  echo "philo: building the index for provider '${PHILO_PROVIDER:-auto}' (one-off)"
  philo ingest --rebuild
fi

exec uvicorn philo.web.app:app --host 0.0.0.0 --port "$PORT" --log-level warning "$@"
