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
#
#    Check free disk BEFORE allocating. The obvious `fallocate || dd` fallback
#    is actively dangerous on a nearly-full root volume: fallocate fails fast
#    and cleanly, but dd then writes zeros until the disk is 100% full, which
#    breaks far more than the install it was trying to rescue. So: refuse up
#    front if the space is not there, and delete any partial file on failure.
# --------------------------------------------------------------------------
SWAP_GB="${SWAP_GB:-2}"
SWAP_HEADROOM_MB=512     # never take the volume to completely full

make_swapfile() {
  local gb="$1"
  if fallocate -l "${gb}G" /swapfile 2>/dev/null; then
    return 0
  fi
  # fallocate is unsupported on some filesystems (older ext3, ZFS), which is a
  # different failure from "no room" -- and one dd genuinely can rescue, now
  # that free space has already been checked.
  log "fallocate unsupported here, falling back to dd"
  dd if=/dev/zero of=/swapfile bs=1M count=$((gb * 1024)) status=none
}

if swapon --show=NAME --noheadings 2>/dev/null | grep -q .; then
  log "swap already active, skipping ($(free -h | awk '/Swap:/ {print $2}') total)"
elif [[ -f /swapfile ]]; then
  log "/swapfile exists but is not active; leaving it alone. Remove it and"
  log "  re-run if you want it rebuilt:  sudo rm -f /swapfile"
else
  avail_mb=$(df -Pm / | awk 'NR==2 {print $4}')
  need_mb=$((SWAP_GB * 1024 + SWAP_HEADROOM_MB))
  if (( avail_mb < need_mb )); then
    die "$(cat <<MSG
only ${avail_mb} MB free on / but a ${SWAP_GB}G swapfile needs ${need_mb} MB
(including ${SWAP_HEADROOM_MB} MB headroom). Nothing was written.

Pick one:
  1. Grow the volume. 8 GB is too small for Gateway + this app + swap; 30 GB
     is still free-tier eligible. Modify the EBS volume in the console, then:
       lsblk
       sudo growpart /dev/nvme0n1 1     # use the device lsblk shows
       sudo resize2fs /dev/nvme0n1p1    # or: sudo xfs_growfs /
  2. Reclaim space:
       docker system df                 # usually the culprit
       docker system prune -a
       sudo apt-get clean && sudo journalctl --vacuum-size=100M
  3. Use a smaller swapfile, thinner than ideal but better than none:
       SWAP_GB=1 sudo -E ./deploy/install.sh
MSG
)"
  fi

  log "creating a ${SWAP_GB}G swapfile (${avail_mb} MB free on /)"
  if ! make_swapfile "$SWAP_GB"; then
    rm -f /swapfile
    die "could not allocate /swapfile; the partial file has been removed"
  fi
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Prefer reclaiming page cache over swapping the Gateway out from under itself.
  sysctl -w vm.swappiness=10 >/dev/null
  grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
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

# ib_async requires Python 3.10+. Amazon Linux 2 ships 3.7, so check here and
# say so plainly rather than letting pip fail with a resolver error 40 lines
# into the install.
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  die "python3 is $PY_VER but ib_async needs 3.10+. On Amazon Linux 2 install python3.11 \
(sudo dnf install python3.11) and re-run as: PYTHON=python3.11 sudo -E ./deploy/install.sh"
fi
log "python3 is $PY_VER"

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
PYTHON="${PYTHON:-python3}"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  log "creating virtualenv with $PYTHON"
  "$PYTHON" -m venv "$APP_DIR/.venv"
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
