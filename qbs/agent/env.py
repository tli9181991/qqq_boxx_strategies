"""Load API keys from a `.env` file, without a dependency and without surprises.

Why a parser and not python-dotenv
-----------------------------------
Because this file is optional infrastructure for an optional feature, and a
`.env` that loads on one machine and not another is worse than one that never
loads. python-dotenv may or may not be installed next to you; a fifty-line
parser is here either way, so the behaviour is the same everywhere.

The cost is a deliberately small syntax. Supported:

    KEY=value
    export KEY=value                 # the `export` prefix is ignored
    KEY="value with spaces"          # matching quotes are stripped
    KEY='value'
    # comments, and blank lines

Not supported, on purpose: variable interpolation (`$OTHER`), multi-line
values, and escape sequences. Anything needing those should be exported by
your shell instead, where it is one obvious mechanism rather than two subtly
different ones. An inline `#` is NOT a comment -- `KEY=a#b` is the four
characters `a#b`, because an API key containing a hash is far likelier than
a trailing comment on a secret.

Precedence
----------
A real environment variable always wins. `load_env` fills in what is missing
and reports what it skipped, so `GOOGLE_API_KEY=... python -m qbs.agent` does
what you would expect even with a `.env` sitting next to it. Pass
`override=True` only if you mean the opposite.

The kill switches
-----------------
`QBS_DISABLE_ANALYST=1` stops every Gemini call in the package;
`QBS_DISABLE_CHAT=1` stops only the Analyst tab's chat and leaves the news
read running. Both live here rather than in `analyst.py` because they are
read the same way as the keys, from the environment or a `.env`, and
`--check` reports them together. See `analyst_disabled` and `chat_disabled`.

Secrets
-------
Nothing here returns, logs or renders a value -- only key names and where
they came from. `.env` is gitignored, and `load_env` warns when the file is
readable by anyone but you.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# The repository root: qbs/agent/env.py -> qbs/agent -> qbs -> root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The kill switches. `QBS_DISABLE_ANALYST` is the master -- set it and no
# Gemini call is made from anywhere in this package. `QBS_DISABLE_CHAT` is
# narrower: it stops the Analyst tab's chat while leaving the news read
# working, which is what you want when testing one feature at a time or when
# the chat is the expensive half and the daily summary is not.
DISABLE_VAR = "QBS_DISABLE_ANALYST"
DISABLE_CHAT_VAR = "QBS_DISABLE_CHAT"

# The keys this package looks for. Listed so `--check` can report on all of
# them, and so a typo in a `.env` can be pointed out rather than ignored.
KNOWN_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY",
              "QBS_GEMINI_MODEL", "QBS_SUMMARY_MODEL", "QBS_THINKING_BUDGET",
              "QBS_DASH_WATCHLIST", DISABLE_VAR, DISABLE_CHAT_VAR)

# Values that mean "switch is off, carry on". Everything else non-empty
# disables -- see `analyst_disabled` for why this is not `_env_bool`.
OFF_VALUES = ("0", "false", "no", "off", "none", "disabled")


@dataclass
class EnvLoad:
    """What a `.env` load did. Key NAMES only -- never a value.

    `error` is fatal: nothing was read. A bad LINE is a `warning` instead,
    because the other lines were still applied, and reporting a whole file as
    failed when the key it carries is now set would send someone hunting for
    the wrong problem.
    """
    path: Optional[str] = None
    applied: List[str] = field(default_factory=list)      # set into os.environ
    skipped: List[str] = field(default_factory=list)      # already in the env
    unknown: List[str] = field(default_factory=list)      # not a KNOWN_KEY
    error: Optional[str] = None                           # fatal: nothing read
    warnings: List[str] = field(default_factory=list)     # read, with caveats
    insecure: bool = False        # readable by group or other

    @property
    def loaded(self) -> bool:
        return self.path is not None and self.error is None

    def summary(self) -> str:
        if self.error:
            return f"{self._where()}: {self.error}"
        if not self.path:
            return "no .env file found"
        bits = [f"loaded {self._where()}"]
        if self.applied:
            bits.append("set " + ", ".join(self.applied))
        if self.skipped:
            bits.append("kept the shell's " + ", ".join(self.skipped))
        bits += self.warnings
        if self.insecure:
            bits.append("WARNING: readable by others — chmod 600 it")
        return "; ".join(bits)

    def _where(self) -> str:
        if not self.path:
            return ".env"
        try:
            rel = os.path.relpath(self.path, REPO_ROOT)
        except ValueError:                  # different drive on Windows
            return self.path
        return self.path if rel.startswith("..") else rel


def parse_env(text: str) -> Tuple[Dict[str, str], List[int]]:
    """Parse `.env` text. Returns `(values, bad_line_numbers)`.

    Lines that are not `KEY=VALUE` are reported rather than skipped silently:
    a typo in the one file holding your API key should not present as "the
    key is missing".
    """
    values: Dict[str, str] = {}
    bad: List[int] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            bad.append(n)
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            bad.append(n)
            continue
        value = value.strip()
        # Strip ONE matching pair of quotes. Anything else, including a `#`,
        # is part of the value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values, bad


def find_env_file(start: Optional[str] = None) -> Optional[str]:
    """The `.env` to use: the repo's own, else one in the working directory.

    No upward walk. A file two directories above the repo belongs to some
    other project, and silently reading a stranger's secrets is not a
    convenience.
    """
    candidates = [os.path.join(REPO_ROOT, ".env")]
    cwd = os.path.abspath(start or os.getcwd())
    if cwd != REPO_ROOT:
        candidates.append(os.path.join(cwd, ".env"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


_CACHE: Dict[str, EnvLoad] = {}


def load_env(path: Optional[str] = None, override: bool = False,
             environ: Optional[Dict[str, str]] = None,
             force: bool = False) -> EnvLoad:
    """Read a `.env` into the environment. Never raises.

    The result is cached per file and replayed on later calls, which is what
    makes the accounting honest. Without it the second call sees the values
    the FIRST call installed, decides the shell must have set them, and
    reports "kept the shell's GOOGLE_API_KEY" about a key that came out of
    the file -- exactly backwards, and precisely the thing someone runs
    `--check` to find out. Pass `force=True` after editing the file.

    `environ` is injectable so a test can verify precedence without touching
    the real process environment; an injected one is never cached.
    """
    env = os.environ if environ is None else environ
    real_env = environ is None
    cache_key = os.path.abspath(path) if path else "<auto>"
    if real_env and not force and cache_key in _CACHE:
        return _CACHE[cache_key]
    def _done(result: EnvLoad) -> EnvLoad:
        if real_env:
            _CACHE[cache_key] = result
        return result

    path = path or find_env_file()
    if path is None:
        return _done(EnvLoad())
    if not os.path.isfile(path):
        return _done(EnvLoad(path=path, error="no such file"))

    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return _done(EnvLoad(path=path, error=f"could not read it ({exc})"))

    values, bad = parse_env(text)
    out = EnvLoad(path=path)
    if bad:
        out.warnings.append(
            f"ignored unparseable line{'s' if len(bad) > 1 else ''} "
            f"{', '.join(map(str, bad))} (expected KEY=value)")

    for key, value in values.items():
        if key not in KNOWN_KEYS:
            out.unknown.append(key)
        if key in env and env[key] and not override:
            out.skipped.append(key)
            continue
        env[key] = value
        out.applied.append(key)

    if out.unknown:
        out.warnings.append("unrecognised key" + ("s " if len(out.unknown) > 1
                                                  else " ")
                            + ", ".join(out.unknown)
                            + " (typo? this package reads "
                            + ", ".join(KNOWN_KEYS) + ")")
    try:
        mode = os.stat(path).st_mode
        out.insecure = bool(mode & (stat.S_IRGRP | stat.S_IROTH))
    except OSError:
        pass
    return _done(out)


def analyst_disabled(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why the analyst is switched off, or None if it is not.

    `QBS_DISABLE_ANALYST=1` stops every Gemini call in this package: the model
    is never constructed, so nothing reaches the API and nothing is billed.
    Everything that does not need the model keeps working -- the picks, the
    momentum profile, breadth, fundamentals, search, `--report`. That is the
    point of a switch rather than an uninstall.

    Deliberately NOT `qbs.live.config._env_bool`, which treats anything
    outside ("1", "true", "yes", "on") as false. For a flag that exists to
    stop spending money, an unrecognised value must fail SAFE: only an
    explicit off-word re-enables, so `QBS_DISABLE_ANALYST=disable` -- a
    perfectly natural thing to type -- switches it off rather than quietly
    leaving it on.

    Returns a sentence, not a bool, because every caller needs to tell
    somebody why: a blank panel with no reason is the thing this avoids.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(DISABLE_VAR) or "").strip()
    if not raw or raw.lower() in OFF_VALUES:
        return None
    return (f"the analyst is switched off by {DISABLE_VAR}={raw!r}. "
            f"Unset it (or set it to 0) to turn Gemini back on; everything "
            f"that does not call the model is unaffected.")


def chat_disabled(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why the Analyst tab's CHAT is switched off, or None if it is not.

    The master switch wins and is reported as the reason: someone who set
    `QBS_DISABLE_ANALYST` needs to be told that, not handed a message about a
    chat switch they never touched.

    Same fail-safe reading as `analyst_disabled` -- only an explicit off-word
    re-enables, so an unrecognised value stops the spending rather than
    quietly leaving it on.
    """
    env = os.environ if environ is None else environ
    master = analyst_disabled(env)
    if master:
        return master
    raw = (env.get(DISABLE_CHAT_VAR) or "").strip()
    if not raw or raw.lower() in OFF_VALUES:
        return None
    return (f"the chat is switched off by {DISABLE_CHAT_VAR}={raw!r}. "
            f"Unset it (or set it to 0) to turn it back on; the news "
            f"sentiment read is unaffected and still runs.")


def resolve_google_key(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """`GOOGLE_API_KEY`, or `GEMINI_API_KEY` if that is what was set.

    Google's own SDKs accept both names and people copy whichever the docs
    they read used. Accepting one and silently ignoring the other produces a
    "key is not set" message while the key is sitting right there in the file.
    """
    env = os.environ if environ is None else environ
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = env.get(name)
        if value:
            return value
    return None
