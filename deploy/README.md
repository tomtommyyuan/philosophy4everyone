# Deploying the web interface

The app is a normal ASGI application (`philo.web.app:app`). Anything that can
run ASGI can host it. The only real constraint is the index.

## The index is the whole problem

`philo ask` needs a built index — a vector matrix plus the chunk metadata. It
is produced by `philo ingest`, it is a few tens of megabytes, and it must be
present before the first request. Where that index lives is what separates an
easy deployment from an awkward one.

| Host | How the index gets there | Verdict |
|---|---|---|
| **Container** (Fly, Render, Railway, Cloud Run) | Built into the image, or built once into a mounted volume on first boot | **Recommended.** Stateful workload, stateful host. |
| **Vercel / serverless** | Must be *committed* and bundled with the function — there is no persistent disk and no build-time volume | Works, with caveats below. |
| **Your own machine** | `philo ingest` | `philo serve` |

## Container hosts (recommended)

```bash
docker build -t philo .
docker run --rm -p 8000:8000 philo                       # offline demo, no key
```

With a real model, mount a volume so the one-off re-index survives restarts:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-… \
  -e PHILO_WEB_TOKEN="$(openssl rand -hex 16)" \
  -v philo-data:/data \
  philo
```

The entrypoint notices that the baked-in offline index does not match the
configured provider and rebuilds once before serving. Fly.io and Render both
read `PORT`, which the image honours.

## Vercel

Serverless functions have a read-only filesystem, so the index has to ship
*inside* the deployment:

```bash
make deploy-index      # builds deploy/index from library/
git add -f deploy/index && git commit -m "chore: prebuilt index for deploy"
vercel --prod
```

`vercel.json` sets `PHILO_INDEX=deploy/index` and bundles it via
`includeFiles`. Set `OPENAI_API_KEY` and `PHILO_WEB_TOKEN` as project
environment variables in the Vercel dashboard.

**Read this before choosing Vercel:**

- **The index must match the provider.** An index built with the mock
  provider will be refused when the function runs against OpenAI. Build
  `deploy/index` with the *same* provider the deployment uses.
- **Bundle size.** OpenAI's `text-embedding-3-small` produces 1536-dim
  vectors: roughly 29 MB of matrix plus ~4 MB of chunk metadata for the
  fourteen-work library. That fits Vercel's limit but is not nothing. The
  offline mock index is ~11 MB.
- **Cold starts.** Loading the matrix and building the BM25 table costs a few
  hundred milliseconds on every cold invocation.
- **Timeouts.** A chat completion plus retrieval can exceed the Hobby plan's
  limit. `maxDuration` is set to 60s, which requires a paid plan.
- **Streaming may be buffered** by the platform. The page falls back to the
  non-streaming `/api/ask` endpoint automatically, so it still works.

## Before you expose it to the internet

**A public URL spends your API credits on behalf of whoever finds it.** There
is no per-user metering in this app.

Set `PHILO_WEB_TOKEN` to a random string. Every `/api/ask`, `/api/daily` and
`/api/search` request then has to present it as an `X-Philo-Token` header;
the web page shows a field for it and remembers it locally.

```bash
export PHILO_WEB_TOKEN="$(openssl rand -hex 16)"
```

Running `philo serve` on `127.0.0.1` needs no token — that is your own
machine. `philo serve --host 0.0.0.0` warns you if the token is unset.

Other things worth doing before a public deployment: put it behind your
host's rate limiting, keep `PHILO_MAX_TOKENS` modest, and consider leaving
`PHILO_PROVIDER=mock` for a demo that cannot cost anything at all.
