#!/usr/bin/env python3
"""Connection smoke test. Works from the host AND from inside the container.

    # from the host, against the published port
    python3 deploy/docker/test_ib.py

    # from inside the compose network, against the Gateway's hostname
    docker compose -f deploy/docker/docker-compose.yml run --rm \
      --entrypoint python qbs test_ib.py

The only change from the original: host and port come from the environment
instead of being hardcoded to 127.0.0.1. Inside the compose network there is no
Gateway on localhost -- it is a separate container reachable as `ib-gateway` --
so a hardcoded loopback address passes on the host and then fails in the very
place it needs to work.
"""

import logging
import os
import sys

from ib_async import IB, Stock

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

HOST = os.environ.get("QBS_IB_HOST", os.environ.get("IB_HOST", "127.0.0.1"))
PORT = int(os.environ.get("QBS_IB_PORT", os.environ.get("IB_PORT", "4002")))
# A client id the scheduled phases do not use, so a test run can never collide
# with a live one -- IB drops the older session when two clients share an id.
CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "99"))

ib = IB()
try:
    print(f"Connecting to IB Gateway at {HOST}:{PORT} (clientId={CLIENT_ID})...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=30)

    accounts = ib.managedAccounts()
    print(f"\nConnected. Managed accounts: {accounts}")

    nlv = next((v.value for v in ib.accountValues()
                if v.tag == "NetLiquidation" and v.currency == "USD"), "n/a")
    print(f"NetLiquidation (USD): {nlv}")

    # Paper accounts start with DU. Worth checking explicitly: connecting to a
    # live account by mistake is the one error here with real consequences.
    if accounts and not any(a.startswith("DU") for a in accounts):
        print(f"\n  WARNING: {accounts} does not look like a paper account "
              f"(paper accounts start with DU). Check TRADING_MODE.")

    print("\nTesting contract details (AAPL)...")
    details = ib.reqContractDetails(Stock("AAPL", "SMART", "USD"))
    print(f"Retrieved contract: {details[0].contract.symbol} "
          f"(conId {details[0].contract.conId})")

    positions = {p.contract.symbol: p.position for p in ib.positions()
                 if p.contract.secType == "STK"}
    print(f"\nCurrent stock positions: {positions or '(flat)'}")
    print("\nOK -- the trader can reach this Gateway.")

except Exception as exc:  # noqa: BLE001
    print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(
        "\nCommon causes:\n"
        f"  - Gateway not up yet          docker compose logs -f ib-gateway\n"
        f"  - wrong host from a container use ib-gateway, not 127.0.0.1\n"
        f"  - API not enabled             READ_ONLY_API=no, and check the\n"
        f"                                Gateway's API settings over VNC\n"
        f"  - clientId already in use     pick another IB_CLIENT_ID",
        file=sys.stderr)
    sys.exit(1)
finally:
    ib.disconnect()
    print("Disconnected.")
