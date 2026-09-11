# Setting up the paper trader on the AWS VM (Docker)

Step by step, from a box that already has Docker and the `gnzsnz/ib-gateway`
image to a scheduled paper trader.

## First: three things about your compose file

Your `dockercompose.yml` will not work as written, for reasons worth
understanding before you fix them.

**1. `command: bash -c "pip install -r requirements.txt && python main.py"`**

There is no `main.py` in this repo, and there should not be. The app is
**three short-lived phases**, not a daemon:

```
08:50 ET  preflight   data + signal + broker + order list, sends nothing
15:30 ET  trade       rank, build, submit MOC before the 15:45 cutoff
16:15 ET  reconcile   fills, end-of-day marks, cancel stragglers
```

Each runs to completion and exits. `restart: unless-stopped` on that service
would restart the trade phase every time it finished — which, once you turn dry
run off, means resubmitting orders in a loop. The compose file here puts the
trader behind a `cli` profile so `up` can never start it, and the systemd timers
invoke it with `docker compose run --rm`.

Also: `requirements.txt` pulls JupyterLab, matplotlib and pytest. On a 2 GB
t3.small use `requirements-live.txt`, which is what the Dockerfile installs.

**2. `test_ib.py` connects to `127.0.0.1:4002`**

That works from the host, because the Gateway publishes 4002. It **fails inside
the container**, where there is no Gateway on loopback — it is a separate
container reachable as `ib-gateway`. A hardcoded loopback address passes the
test you run by hand and then fails in the place it actually matters.
`deploy/docker/test_ib.py` reads host and port from the environment so the same
script proves both paths.

**3. `ports: "4002:4002"` and `"5900:5900"` bind to every interface**

That publishes the IB API and VNC to `0.0.0.0`. The API port has no
authentication beyond IB's trusted-IP list; VNC has only a password. If your
security group ever opens those ports, anyone who finds them can trade your
account. The compose file here binds both to `127.0.0.1` and you reach them over
an SSH tunnel.

`version: '3.8'` is also obsolete in Compose v2 and just prints a warning.

---

## Step 1 — Get the code onto the box

```bash
sudo mkdir -p /opt/qbs && sudo chown "$USER:$USER" /opt/qbs
git clone https://github.com/tli9181991/qqq_boxx_strategies.git /opt/qbs
cd /opt/qbs
```

Everything below runs from `/opt/qbs`.

## Step 2 — Disk, then swap

A t3.small has 2 GB of RAM. The Gateway's JVM takes most of 1 GB, and the trade
phase holds a ~100-ticker × 1500-row price frame. It fits, but without swap the
OOM killer will eventually take the Python process mid-session — and it does
that silently. You would find out from a missing auction, not an error.

**Check the disk first.** A 2 GB swapfile needs 2 GB of free disk, and the
default 8 GB root volume plus the `ib-gateway` image (1–2 GB) does not leave it:

```bash
df -h /
docker system df        # images are usually what filled it
```

If free space is under ~2.5 GB, deal with that before going further:

```bash
# Best: grow the volume. 8 GB is too small for Gateway + this app + swap.
# 30 GB is still free-tier eligible. Modify the EBS volume in the console, then
# find your device with lsblk -- the name depends on the instance family:
#   Nitro (t3, t4g, m5...):  /dev/nvme0n1  partition /dev/nvme0n1p1
#   Xen   (t2, m4...):       /dev/xvda     partition /dev/xvda1
lsblk
sudo growpart /dev/nvme0n1 1      # <-- substitute what lsblk shows
sudo resize2fs /dev/nvme0n1p1     # <-- likewise; or: sudo xfs_growfs /
df -h /

# Or reclaim:
docker system prune -a
sudo apt-get clean && sudo journalctl --vacuum-size=100M
```

Then create the swapfile:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
free -h        # confirm 2.0Gi of swap
```

If `fallocate` fails with **"No space left on device"**, stop and fix the disk —
and check for a partial file first, because a half-written swapfile is holding
space you need: `sudo swapoff /swapfile 2>/dev/null; sudo rm -f /swapfile`.

> **Do not run `deploy/install.sh`.** That is the native install path. It builds
> a venv and installs systemd units with the *same names* as the Docker ones, so
> whichever you install last silently wins. Everything you need is on this page.

## Step 3 — Fill in the environment file

```bash
cp deploy/docker/.env.example deploy/docker/.env
chmod 600 deploy/docker/.env
echo "QBS_UID=$(id -u)"; echo "QBS_GID=$(id -g)"     # put these in the file
nano deploy/docker/.env
```

Set `TWS_USERID`, `TWS_PASSWORD`, `VNC_PASSWORD`, and the `QBS_UID`/`QBS_GID`
you just printed. The UID matters: without it the container writes the run log
as root into your repo, and you then need sudo to read your own trade history.

**Leave `QBS_DRY_RUN=1`.** Step 9 turns it off.

`deploy/docker/.env` is gitignored — check with `git status` that it does not
appear before you ever push.

## Step 4 — Start the Gateway

Clear out anything left from an earlier attempt first. This matters more than it
looks: `container_name: ib-gateway` is fixed in the compose file, so a container
of that name from a previous run blocks the new one with *"container name
already in use"*. And `docker compose down` will **not** remove it if it came
from a different compose file — Compose only touches containers carrying its own
project label — so the removal has to be by name.

```bash
docker ps -a                                          # look before you delete
docker compose -f deploy/docker/docker-compose.yml down --remove-orphans
docker rm -f ib-gateway trading-bot 2>/dev/null || true
```

If this VM is dedicated to the trader and you would rather start from nothing:

```bash
docker rm -f $(docker ps -aq) 2>/dev/null || true     # removes EVERY container
```

Only on a box you are sure about — that takes out containers belonging to
anything else running here. It removes containers, not images, so nothing has to
be re-pulled.

Now start it:

```bash
docker compose -f deploy/docker/docker-compose.yml up -d ib-gateway
docker compose -f deploy/docker/docker-compose.yml logs -f ib-gateway
```

Wait for the login to complete — first start pulls the image and can take a few
minutes. Watch for the API listening on 4002. `Ctrl-C` stops following the log,
not the container.

If login fails, it is almost always the credentials or `TRADING_MODE`. Look at
the Gateway's own screen over VNC (step 5) rather than guessing.

## Step 5 — Watch the Gateway's screen, if you need to

VNC is bound to localhost, so tunnel to it. **From your laptop:**

```bash
ssh -N -L 5900:127.0.0.1:5900 ubuntu@YOUR_VM_IP
```

Then point a VNC client at `localhost:5900` with your `VNC_PASSWORD`. Use this
to confirm the API is enabled and to clear any dialog the Gateway is stuck on.
Close the tunnel when you are done.

## Step 6 — Prove the connection, both ways

From the host first — this is the simpler failure to diagnose:

```bash
pip install --user ib_async
python3 deploy/docker/test_ib.py
```

Then from inside the compose network, which is what the trader actually does:

```bash
docker compose -f deploy/docker/docker-compose.yml build
docker compose -f deploy/docker/docker-compose.yml run --rm \
  --entrypoint python qbs test_ib.py
```

Both must print `Connected` and a `DU…` account number. The script warns if the
account does not look like a paper account.

**If the host test passes and the container test fails**, it is the hostname:
inside the network the Gateway is `ib-gateway`, never `127.0.0.1`.

## Step 7 — Check the signal, with no broker involved

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm qbs signal --offline
```

This touches neither the network nor IB — it ranks the cached price history and
prints today's target book. If this fails, the problem is the app, not the
plumbing, and you have separated the two.

Then the live-data version:

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm qbs signal
```

## Step 8 — Full preflight

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm qbs preflight
```

This is the whole path: downloads, ranks, connects to IB, reads your positions,
builds the exact order list, and **sends nothing**. Read the order table it
prints. That list is what the trade phase would submit.

Confirm the run log was written and is yours, not root's:

```bash
ls -l var/
docker compose -f deploy/docker/docker-compose.yml run --rm qbs report
```

## Step 9 — Install the timers

```bash
sudo cp deploy/docker/systemd/qbs-*.service deploy/docker/systemd/qbs-*.timer \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qbs-preflight.timer qbs-trade.timer qbs-reconcile.timer
systemctl list-timers 'qbs-*'
```

`list-timers` should show the next run in **UTC** corresponding to 08:50 / 15:30
/ 16:15 New York. Check the arithmetic once — during EDT that is 12:50 / 19:30 /
20:15 UTC. If they show as 08:50 UTC, `tzdata` is missing on the host.

The timers now run with `QBS_DRY_RUN=1` from your `.env`. **Leave it that way
for a few sessions.** Compare each day's order list against the backtest:

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm qbs report --days 10
```

When the order lists stop surprising you, set `QBS_DRY_RUN=0` in
`deploy/docker/.env`. No restart is needed — each phase reads the file when it
runs.

## Step 10 — The instance schedule

Two EventBridge Scheduler rules against the EC2 API, both in
`America/New_York` so they track DST with the timers:

```
qbs-start   cron(45 8 ? * MON-FRI *)    ec2:StartInstances
qbs-stop    cron(20 16 ? * MON-FRI *)   ec2:StopInstances
```

Make sure the Gateway container comes back on boot — `restart: unless-stopped`
handles that only if the Docker service is enabled:

```bash
sudo systemctl enable docker
```

Neither rule knows about market holidays. That is harmless: the trade phase sees
no bar for today and exits 0 without trading. You pay for idle hours, not for a
bad trade.

---

## Operating it

```bash
cd /opt/qbs
C="docker compose -f deploy/docker/docker-compose.yml"

$C ps                                    # is the Gateway up
$C logs --tail 50 ib-gateway
$C run --rm qbs report --days 20         # trades, picks, closes
$C run --rm qbs signal --offline         # what would it hold today

systemctl list-timers 'qbs-*'
journalctl -u qbs-trade -n 100 --no-pager
sudo systemctl start qbs-preflight.service   # force a phase now
```

### Kill switch

```bash
touch /opt/qbs/var/HALT      # preflight and trade refuse; reconcile still runs
rm /opt/qbs/var/HALT         # resume
```

A missed session costs nothing — the strategy recomputes from full price history
every run, so there is never anything to catch up.

### Updating the code

```bash
cd /opt/qbs && git pull
docker compose -f deploy/docker/docker-compose.yml build qbs
```

The Gateway container is untouched by this.

---

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionRefused` from the container | wrong host | it is `ib-gateway`, not `127.0.0.1` |
| `ConnectionRefused` from the host | Gateway not up or still logging in | `$C logs ib-gateway`; first start takes minutes |
| Connects, then drops | two clients sharing an id | phases use 17, the test uses 99 — keep them apart |
| Account is not `DU…` | `TRADING_MODE` not `paper` | fix `.env`, recreate the Gateway container |
| `var/` files owned by root | `QBS_UID`/`QBS_GID` unset | set them, then `sudo chown -R $(id -u):$(id -g) var/` |
| Timers fire at the wrong hour | no `tzdata` on the host | `sudo apt-get install tzdata`, `daemon-reload` |
| Trade phase: "past the 15:45 MOC cutoff" | the run took too long | usually the yfinance download; nothing to do today |
| Trade phase exits 3, "GUARD TRIPPED" | an order list breached a limit; **nothing was sent** | read the reason — a turnover breach is nearly always a bad universe download |
| OOM / container killed | no swap | step 2 |

Exit codes: `0` fine (including "market closed today"), `1` error, `2` config or
connection, `3` a guard refused the list.

---

## What this deliberately does not do

- **No healthcheck on the Gateway container.** `depends_on` here is start-order
  only. A `condition: service_healthy` gate would be better, but it needs a
  probe whose tooling I could not verify inside that image — add one once you
  have looked at what the image ships.
- **No alerting.** Failures surface as failed systemd units. Wire `OnFailure=`
  to something that emails you if you want to hear before the next morning.
- **The trader container has no IB credentials.** It talks to the Gateway over
  the network and never needs the login, so `.env` is not passed to it.
