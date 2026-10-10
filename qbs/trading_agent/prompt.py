"""The Daily User Prompt: what to watch today, and how.

The prompt is natural language and goes to the model as written -- but two
things are pulled out of it HERE, in code, and enforced outside the model:

* **the authorized symbols.** The model never decides what it may talk
  about. A decision for any symbol not on this list is rejected by
  `decision.validate_decision`, whatever the model was persuaded of.
* **explicit constraints** a regex can read without guessing: no short
  selling, a maximum share count, "do not add", an existing position size,
  a reference stop. Each becomes a hard check on the decision. Anything the
  regexes do not recognise stays prose for the model and is never read as
  permission for anything.

Symbol extraction is deliberately strict. The authoritative source is an
explicit list line -- "Monitor only TEAM, MRVL, and TSM." (also "Watchlist:",
"Symbols:", "Tickers:") -- and, only when there is none, the per-stock
section headers ("TEAM:" on a line of its own). A capitalised word in a
sentence ("Watch for a breakout above $200, AI names are hot") never becomes
a symbol. When both exist and disagree, the list wins and the stray header
is reported as a warning, because an ambiguous instruction must not widen
the universe.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Optional

MAX_SYMBOLS = 10        # one LLM call each, every 15 minutes
MAX_PROMPT_CHARS = 8000

SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")
_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}(?:[.-][A-Z]{1,2})?\b")
_LIST_RE = re.compile(
    r"^\s*(?:monitor\s+only\s*:?|(?:monitor|watch\s*list|symbols|tickers|stocks)"
    r"\s*:)\s*(.+)$", re.I)
_HEADER_RE = re.compile(r"^\s*([A-Z]{1,5}(?:[.-][A-Z]{1,2})?)\s*:\s*(.*)$")
_GENERAL_RE = re.compile(r"^\s*(general|notes?|rules|market|overall)\s*:\s*(.*)$", re.I)
_DATE_RE = re.compile(r"^\s*date\s*:\s*(\d{4}-\d{2}-\d{2})\b", re.I | re.M)

# Upper-case words that are not tickers in a list line ("ONLY TEAM AND TSM",
# "TSM ET", "NO ETFs"). A real ticker on this list (e.g. "ON", "IT") must be
# written as a section header to be picked up.
_NOT_TICKERS = {"A", "AND", "OR", "ONLY", "THE", "ET", "EST", "EDT", "NY",
                "US", "USA", "NO", "NOT", "DO", "I", "ON", "IN", "AT", "TO",
                "FOR", "OF", "MY", "ALL", "ETF", "ETFS", "SL", "TP", "RSI",
                "EMA", "SMA", "ATR", "MACD", "VWAP", "AI", "IPO", "PM", "AM",
                "BUY", "SELL", "HOLD", "DATE", "NOTE", "USD"}

_PRICE = r"\$?\s*(\d+(?:\.\d+)?)"


@dataclass
class SymbolConstraints:
    """Hard limits for one symbol, read from the prompt. None = not stated."""
    max_quantity: Optional[int] = None
    no_add: bool = False                   # "do not recommend adding"
    position_qty: Optional[int] = None     # "my existing 9-share position"
    reference_stop: Optional[float] = None
    key_levels: List[float] = field(default_factory=list)


@dataclass
class ParsedPrompt:
    text: str
    sha256: str
    prompt_date: Optional[str]
    symbols: List[str]
    sections: Dict[str, str]
    general: str
    no_short: bool
    constraints: Dict[str, SymbolConstraints]
    # Only an explicit sentence ("Short selling is allowed") sets this, and
    # never alongside a "no short selling" line.
    allow_short: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def instructions_for(self, symbol: str) -> str:
        """The part of the prompt about `symbol`, plus the general rules.

        Sent instead of the whole prompt to save tokens -- other stocks'
        sections are noise for this decision. Falls back to the full text
        when the prompt has no sections, so nothing written is ever lost.
        """
        own = self.sections.get(symbol)
        if own is None and not self.general:
            return self.text.strip()
        parts = [f"Authorized symbols today: {', '.join(self.symbols)}."]
        if own:
            parts.append(f"{symbol}:\n{own}")
        else:
            parts.append(f"{symbol}: (no stock-specific instructions)")
        if self.general:
            parts.append(f"General:\n{self.general}")
        return "\n\n".join(parts)

    def constraints_dict(self) -> Dict[str, Dict]:
        return {s: asdict(c) for s, c in self.constraints.items()}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tickers_in(text: str) -> List[str]:
    out = []
    for tok in _TOKEN_RE.findall(text):
        if tok in _NOT_TICKERS or tok in out:
            continue
        out.append(tok)
    return out


def _constraints(section: str) -> SymbolConstraints:
    c = SymbolConstraints()
    low = section.lower()
    m = re.search(r"max(?:imum)?\s+(?:suggested\s+)?(?:position(?:\s+size)?)?\s*:?\s*"
                  r"(\d+)\s*shares?", low)
    if m:
        c.max_quantity = int(m.group(1))
    if re.search(r"(do not|don't|never)\s+(recommend\s+)?add|no\s+(new|additional)\s+shares|"
                 r"no\s+adding", low):
        c.no_add = True
    m = re.search(r"existing\s+(\d+)[\s-]*shares?\s+position", low) or \
        re.search(r"(?:hold|own|holding|long)\s+(\d+)\s+shares", low)
    if m:
        c.position_qty = int(m.group(1))
    m = re.search(r"stop[\s-]*loss\s*(?:reference)?\s*(?:at|:)?\s*" + _PRICE, low)
    if m:
        c.reference_stop = float(m.group(1))
    c.key_levels = sorted({float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", section)})
    return c


def parse_prompt(text: str, session_date: Optional[date] = None) -> ParsedPrompt:
    """Read a Daily User Prompt. Never raises; problems land in `errors`."""
    text = (text or "").replace("\r\n", "\n")
    errors: List[str] = []
    warnings: List[str] = []
    if not text.strip():
        errors.append("the daily prompt is empty")
    if len(text) > MAX_PROMPT_CHARS:
        errors.append(f"the daily prompt is longer than {MAX_PROMPT_CHARS} characters")

    m = _DATE_RE.search(text)
    prompt_date = m.group(1) if m else None
    if prompt_date and session_date and prompt_date != session_date.isoformat():
        errors.append(f"the prompt is dated {prompt_date} but the session is "
                      f"{session_date.isoformat()}; a previous day's prompt "
                      f"cannot run today")

    listed: List[str] = []
    sections: Dict[str, List[str]] = {}
    general: List[str] = []
    current: Optional[List[str]] = None
    for line in text.split("\n"):
        lm = _LIST_RE.match(line)
        if lm and not listed and _tickers_in(lm.group(1)):
            listed = _tickers_in(lm.group(1))
            current = None
            continue
        hm = _HEADER_RE.match(line)
        if hm and hm.group(1) not in _NOT_TICKERS:
            current = sections.setdefault(hm.group(1), [])
            if hm.group(2).strip():
                current.append(hm.group(2).strip())
            continue
        gm = _GENERAL_RE.match(line)
        if gm:
            current = general
            if gm.group(2).strip():
                general.append(gm.group(2).strip())
            continue
        if current is not None and line.strip():
            current.append(line.strip())

    if listed:
        symbols = listed
        for s in sections:
            if s not in listed:
                warnings.append(f"a section is headed {s}: but {s} is not in the "
                                f"monitor list; it is ignored, not added")
        sections = {s: v for s, v in sections.items() if s in listed}
    else:
        symbols = list(sections)
        if symbols:
            warnings.append("no explicit 'Monitor only ...' line; the symbols "
                            "were taken from the section headers")

    bad = [s for s in symbols if not SYMBOL_RE.match(s)]
    if bad:
        errors.append(f"not valid ticker symbols: {', '.join(bad)}")
    if text.strip() and not symbols:
        errors.append("no symbols found: add a line like "
                      "'Monitor only TEAM, MRVL, and TSM.'")
    if len(symbols) > MAX_SYMBOLS:
        errors.append(f"{len(symbols)} symbols; at most {MAX_SYMBOLS} per session")

    low = text.lower()
    no_short = bool(re.search(r"no\s+short(ing|\s+selling)?|(do not|don't|never)\s+"
                              r"(recommend\s+)?short", low))
    allow_short = (not no_short and bool(re.search(
        r"short(ing|\s+selling)?\s+(is\s+)?(allowed|permitted|ok)|(may|can)\s+short|"
        r"allow(ed)?\s+(to\s+)?short", low)) and not re.search(
        r"short(ing|\s+selling)?\s+(is\s+)?not\s+(allowed|permitted)", low))
    sec_text = {s: "\n".join(v) for s, v in sections.items()}
    constraints = {s: _constraints(sec_text.get(s, "")) for s in symbols}
    return ParsedPrompt(text=text, sha256=sha256(text), prompt_date=prompt_date,
                        symbols=symbols, sections=sec_text,
                        general="\n".join(general), no_short=no_short,
                        allow_short=allow_short,
                        constraints=constraints, errors=errors, warnings=warnings)
