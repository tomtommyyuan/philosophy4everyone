"""Vercel entry point.

Vercel's Python runtime looks for a module-level ASGI `app`. Everything else
lives in `philo.web.app`, so this file stays a one-line adapter and the same
code runs locally under `philo serve`.

Read `deploy/README.md` before deploying — a serverless function has no
persistent disk, so the index has to be bundled with the deployment.
"""

from philo.web.app import app

__all__ = ["app"]
