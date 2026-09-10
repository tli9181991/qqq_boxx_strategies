#!/usr/bin/env bash
#
# Install the QBS live trader on a fresh Ubuntu/Amazon Linux box.
#
#   sudo ./deploy/install.sh
#
# Idempotent: safe to re-run after a git pull. It does NOT start the timers --
# see the runbook, which walks through a dry run first.
#
set -euo pipefail

APP_USER="${APP_USER:-qbs}"
APP_DIR="${APP_DIR:-/opt/qbs}"
ETC_DIR="${ETC_DIR:-/etc/qbs}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run me with sudo"

# --------------------------------------------------------------------------
# 1. Swap. A t3.small has 2 GB, and IB Gateway's JVM wants most of 1 GB of it.
#    Add pandas plus a 100-ticker download and the OOM killer becomes a real
#    risk -- and it will pick the Python process, mid-session, silently.
# --------------------------------------------------------------------------
if [[ ! -f /swapfile ]]; then
  log "creating a 2G swapfile (t3.small has only 2G of RAM)"
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Prefer reclaiming page cache over swapping the Gateway out from under itself.
  sysctl -w vm.swappiness=10 >/dev/null
  grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
else
  log "swapfile already present, skipping"
fi

# --------------------------------------------------------------------------
# 2. Timezone data. The timers are anchored to America/New_York; without tzdata
#    systemd silently falls back to UTC and every job fires at the wrong hour.
# --------------------------------------------------------------------------
log "ensuring tzdata is installed"
if command -v apt-get >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tzdata python3-venv python3-pip
elif command -v dnf >/dev/null; then
  dnf install -y -q tzdata python3-pip
else
  die "unsupported package manager; install tzdata and python3-venv by hand"
fi
[[ -f /usr/share/zoneinfo/America/New_York ]] || die "tzdata did not provide America/New_York"

# --------------------------------------------------------------------------
# 3. Service account and directories
# --------------------------------------------------------------------------
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  log "creating service user $APP_USER"
  useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

log "syncing $SRC_DIR -> $APP_DIR"
mkdir -p "$APP_DIR" "$APP_DIR/var" "$ETC_DIR"
# --delete keeps a removed file from lingering, but never touch var/ (state,
# order and fill logs) or data/ (the price cache).
rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  --exclude 'var/' --exclude 'output/' --exclude 'notebooks/' \
  "$SRC_DIR"/ "$APP_DIR"/

# --------------------------------------------------------------------------
# 4. Virtualenv
# --------------------------------------------------------------------------
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  log "creating virtualenv"
  python3 -m venv "$APP_DIR/.venv"
fi
log "installing dependencies (this is the slow step)"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements-live.txt"

# --------------------------------------------------------------------------
# 5. Config. Never overwrite one that already exists -- it holds the account
#    number and the notional, and clobbering it on a re-run would silently
#    resize the book.
# --------------------------------------------------------------------------
if [[ ! -f "$ETC_DIR/live.json" ]]; then
  log "installing $ETC_DIR/live.json from the example"
  install -m 0640 -o "$APP_USER" -g "$APP_USER" \
    "$APP_DIR/deploy/live.example.json" "$ETC_DIR/live.json"
else
  log "$ETC_DIR/live.json exists, leaving it alone"
fi
if [[ ! -f "$ETC_DIR/live.env" ]]; then
  install -m 0640 -o "$APP_USER" -g "$APP_USER" \
    "$APP_DIR/deploy/live.env.example" "$ETC_DIR/live.env"
fi

chown -R "$APP_USER:$APP_USER" "$APP_DIR" "$ETC_DIR"

# --------------------------------------------------------------------------
# 6. systemd
# --------------------------------------------------------------------------
log "installing systemd units"
install -m 0644 "$APP_DIR"/deploy/systemd/*.service "$APP_DIR"/deploy/systemd/*.timer \
  /etc/systemd/system/
systemctl daemon-reload

cat <<'DONE'

==> Installed. Nothing is scheduled yet, on purpose.

Next, in order:

  1. Check the config:            sudo -u qbs cat /etc/qbs/live.json
  2. Signal only, no broker:      sudo -u qbs /opt/qbs/.venv/bin/python \
                                    -m qbs.live.runner signal
  3. Start IB Gateway and log in to the PAPER account.
  4. Full preflight:              sudo systemctl start qbs-preflight.service
                                  journalctl -u qbs-preflight -n 60 --no-pager
  5. Read the order list it prints. If it looks right, enable the timers:
                                  sudo systemctl enable --now \
                                    qbs-preflight.timer qbs-trade.timer qbs-reconcile.timer
  6. Confirm the schedule:        systemctl list-timers 'qbs-*'

The runbook at /opt/qbs/deploy/README.md covers the AWS start/stop schedule,
the kill switch, and what to do when a phase fails.
DONE
