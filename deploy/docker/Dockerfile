# The live trader. Built from the repo root:
#   docker compose -f deploy/docker/docker-compose.yml build
#
# Deliberately NOT a long-running service. The app is three short-lived phases
# (preflight / trade / reconcile) that each run to completion and exit, so this
# image is invoked with `docker compose run --rm`, never `up`.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/New_York

WORKDIR /app

# tzdata: every schedule in this app is exchange-local (the 15:45 MOC cutoff,
# the "is it a weekday in New York" check). Python's zoneinfo on a slim image
# has no database to read without it, and would silently fall back to UTC --
# which would put the trade phase four or five hours off.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# requirements-live.txt, NOT requirements.txt: the latter pulls JupyterLab,
# matplotlib and pytest, which are a few hundred MB this box does not need and
# has little room for.
COPY requirements-live.txt .
RUN pip install --no-cache-dir -r requirements-live.txt

COPY qbs/ ./qbs/
# run_backtest.py stays so you can compare the live signal against the
# backtest on the same box. Analysis scripts do not belong in a trading
# image -- they pull chart dependencies this box has no room for.
COPY run_backtest.py ./
# The connection smoke test, so it can be run from inside the network -- which
# is the path that actually matters and the one a host-side test cannot prove.
COPY deploy/docker/test_ib.py ./test_ib.py
# The price cache, so `--offline` works even with no volume mounted.
COPY data/ ./data/

# The runner is the entrypoint, so the command is just the phase name:
#   docker compose run --rm qbs preflight
# For anything else, override it:  --entrypoint python
ENTRYPOINT ["python", "-m", "qbs.live.runner"]
CMD ["signal", "--offline"]
