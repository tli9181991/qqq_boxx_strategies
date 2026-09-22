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
Three, one per thing that can spend, and no master above them:

* `QBS_DISABLE_NEWS_ANALYSIS` -- the News tab's Gemini read. **Off by
  default**, so a fresh checkout with a key in it does not start billing;
  `=0` switches it on. It is the only switch here whose default is the
  disabled position, because it is the only one that spends without anybody
  pressing anything.
* `QBS_DISABLE_NEWS_READ` -- fetching the headlines themselves. On by
  default, since that is Tavily credits at worst and free at best. Ignored
  while the analysis is on: there is nothing to analyse without it.
* `QBS_DISABLE_CHAT` -- the Analyst tab's chat, which spends only when
  somebody types.

They live here rather than in `analyst.py` because they are read the same
way as the keys, from the environment or a `.env`, and `--check` reports
them together. See `news_analysis_disabled`, `news_read_disabled` and
`chat_disabled`.

`QBS_DISABLE_ANALYST` was the master switch over all of these and is
retired. A switch that is read by nothing is worse than one that is gone, so
a leftover setting is reported rather than ignored -- see `RETIRED_VARS`.

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

# One switch per thing that can spend. There is deliberately no master above
# them: a master is a second answer to "is this on?", and the two answers
# disagree the moment someone sets only one of them.
DISABLE_CHAT_VAR = "QBS_DISABLE_CHAT"
DISABLE_NEWS_ANALYSIS_VAR = "QBS_DISABLE_NEWS_ANALYSIS"
DISABLE_NEWS_READ_VAR = "QBS_DISABLE_NEWS_READ"

# Retired switches, and what replaced each. Kept so a leftover line can be
# reported instead of silently doing nothing -- a kill switch that has
# quietly stopped being read is the most dangerous kind of dead config, and
# somebody is relying on this one to hold their bill down.
RETIRED_VARS = {
    "QBS_DISABLE_ANALYST": (
        f"it was the master switch over everything. The news read is now "
        f"{DISABLE_NEWS_ANALYSIS_VAR} (off unless you set it to 0) and the "
        f"chat is {DISABLE_CHAT_VAR}. Nothing reads the old name, so this "
        f"line no longer switches anything off."),
}

# The keys this package looks for. Listed so `--check` can report on all of
# them, and so a typo in a `.env` can be pointed out rather than ignored.
KNOWN_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY",
              "QBS_GEMINI_MODEL", "QBS_SUMMARY_MODEL", "QBS_THINKING_BUDGET",
              "QBS_DASH_WATCHLIST", DISABLE_CHAT_VAR,
              DISABLE_NEWS_ANALYSIS_VAR, DISABLE_NEWS_READ_VAR)

# Values that mean "switch is off, carry on". Everything else non-empty
# disables -- see `chat_disabled` for why this is not `_env_bool`.
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
    retired: List[str] = field(default_factory=list)     # a RETIRED_VAR
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
        if key in RETIRED_VARS:
            # Named separately from the unknown keys, and not as a typo: this
            # one WAS right, and the remedy is "here is what replaced it"
            # rather than "check your spelling". Still applied to the
            # environment, so anything outside this package that reads it
            # keeps working.
            out.retired.append(key)
        elif key not in KNOWN_KEYS:
            out.unknown.append(key)
        if key in env and env[key] and not override:
            out.skipped.append(key)
            continue
        env[key] = value
        out.applied.append(key)

    for key in out.retired:
        out.warnings.append(f"{key} is RETIRED and no longer read — "
                            + RETIRED_VARS[key])
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


def _off_word(raw: str) -> bool:
    """Does this value read as "switch is off, carry on"?"""
    return not raw or raw.strip().lower() in OFF_VALUES


def news_analysis_disabled(environ: Optional[Dict[str, str]] = None
                           ) -> Optional[str]:
    """Why the News tab's Gemini read is off, or None if it is on.

    **The only switch here that defaults to the disabled position.** Unset
    means off. Every other switch in this file guards something a person
    starts -- typing in the chat, pressing refresh -- so leaving it on until
    told otherwise costs nothing until somebody acts. This one runs on page
    load, once a day, for anybody who happens to have a key in their `.env`,
    and a default that bills a new checkout for opening the dashboard is not
    a default anybody chose.

    So it is switched on explicitly: `QBS_DISABLE_NEWS_ANALYSIS=0`. Anything
    else -- unset, "1", "yes", a typo -- leaves it off. The headlines are
    unaffected either way; see `news_read_disabled`.

    Returns a sentence, not a bool, because every caller needs to tell
    somebody why: a blank panel with no reason is the thing this avoids. The
    sentence distinguishes "never switched on" from "switched off", because
    the remedies read the same but the surprise is completely different.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(DISABLE_NEWS_ANALYSIS_VAR) or "").strip()
    if raw and _off_word(raw):
        return None
    if not raw:
        return (f"the news analysis is off by default — it is the one thing "
                f"here that would spend on its own. Set "
                f"{DISABLE_NEWS_ANALYSIS_VAR}=0 in your .env to switch the "
                f"Gemini read on. The headlines below do not need it.")
    return (f"the news analysis is switched off by "
            f"{DISABLE_NEWS_ANALYSIS_VAR}={raw!r}. Set it to 0 to switch the "
            f"Gemini read on; the headlines are unaffected either way.")


def news_read_disabled(environ: Optional[Dict[str, str]] = None
                       ) -> Optional[str]:
    """Why the headline fetch is off, or None if it is on.

    On unless switched off, the opposite of `news_analysis_disabled`: this
    is a search, free on DuckDuckGo and a credit on Tavily, not an LLM call.

    **Switched-on analysis wins.** Asking for a read of the news while the
    news is not being fetched is not a configuration anybody means; it is
    two settings that contradict each other, and the one someone went out of
    their way to enable is the one that says what they wanted. So the fetch
    runs, and this returns None.

    Same fail-safe reading as the others -- only an explicit off-word keeps
    it running, so an unrecognised value stops the fetching rather than
    quietly leaving it on.
    """
    env = os.environ if environ is None else environ
    if news_analysis_disabled(env) is None:
        return None
    raw = (env.get(DISABLE_NEWS_READ_VAR) or "").strip()
    if _off_word(raw):
        return None
    return (f"the news fetch is switched off by "
            f"{DISABLE_NEWS_READ_VAR}={raw!r}. Unset it (or set it to 0) to "
            f"fetch headlines again. Switching the analysis on overrides "
            f"this, since there is nothing to analyse without it.")


def chat_disabled(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why the Analyst tab's CHAT is switched off, or None if it is not.

    On unless switched off: the chat spends only when somebody types into
    it, so there is nothing for a default to protect them from.

    Deliberately NOT `qbs.live.config._env_bool`, which treats anything
    outside ("1", "true", "yes", "on") as false. For a flag that exists to
    stop spending money, an unrecognised value must fail SAFE: only an
    explicit off-word re-enables, so `QBS_DISABLE_CHAT=disable` -- a
    perfectly natural thing to type -- switches it off rather than quietly
    leaving it on.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(DISABLE_CHAT_VAR) or "").strip()
    if _off_word(raw):
        return None
    return (f"the chat is switched off by {DISABLE_CHAT_VAR}={raw!r}. "
            f"Unset it (or set it to 0) to turn it back on; the news "
            f"sentiment read answers to its own switch and is unaffected.")


def role_disabled(role: str = "chat",
                  environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The switch that applies to `role`: "chat" or "summary".

    One mapping, used by `check_requirements`, `build_model` and the
    dashboard alike. Three copies of `if role == "chat"` is how a role ends
    up checked against the wrong switch -- which, for switches that exist to
    stop spending, means the one that was set is the one that gets ignored.
    """
    return (chat_disabled(environ) if role == "chat"
            else news_analysis_disabled(environ))


def retired_vars_in_use(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """Retired switch names that are currently set to something.

    Reported rather than ignored. Somebody who set a kill switch is relying
    on it, and the failure mode of a silently-retired one is a bill.
    """
    env = os.environ if environ is None else environ
    return [k for k in RETIRED_VARS if (env.get(k) or "").strip()]


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
