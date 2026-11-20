"""Fetch a web page and extract its readable text — standard library only.

Search engines return a title, a URL and a ~200-character snippet. A snippet rarely
contains the number or date a claim needs, so the summarizer would either cite thin
evidence or fill gaps from memory (an unsupported claim). Fetching the page gives the
pipeline the actual text; ``budget.select_passages`` then keeps only the parts of it
that are relevant to the sub-question.

Extraction is deliberately simple: drop script/style/nav/header/footer/aside/forms,
keep text from block elements, and discard short menu-like lines.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; research-pipeline/0.1)"

_SKIP = frozenset(
    "script style noscript nav header footer aside form svg iframe template button select".split()
)
_BLOCK = frozenset(
    "p div section article main li ul ol h1 h2 h3 h4 h5 h6 tr td th br blockquote pre "
    "table dd dt figcaption".split()
)
_VOID = frozenset("br hr img input meta link source wbr area base col embed track".split())
_SPACES = re.compile(r"\s+")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.title: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK or tag in _VOID:  # <br/> etc. never open a skipped region
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
        elif not self.skip_depth:
            self.parts.append(data)


def _keep(line: str) -> bool:
    """Menus and buttons are short; sentences are long or end with punctuation."""
    return len(line) >= 60 or (len(line) >= 20 and line[-1] in ".!?:;。！？")


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text). Paragraphs are separated by blank lines."""
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep whatever was parsed  # noqa: BLE001
        pass
    seen: set[str] = set()
    lines: list[str] = []
    for raw in "".join(parser.parts).split("\n"):
        line = _SPACES.sub(" ", raw).strip()
        if _keep(line) and line not in seen:
            seen.add(line)
            lines.append(line)
    title = _SPACES.sub(" ", "".join(parser.title)).strip()
    return title, "\n\n".join(lines)


async def fetch_page_text(
    http: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 8.0,  # noqa: ASYNC109 - per-request httpx timeout, not a deadline
    max_chars: int = 20_000,
    max_bytes: int = 2_000_000,
    min_chars: int = 200,
) -> str | None:
    """Download ``url`` and return its readable text, or None if that is not possible.

    Never raises for network/HTTP problems: a page that cannot be fetched simply keeps
    its search snippet. Cancellation still propagates.
    """
    try:
        async with http.stream(
            "GET",
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        ) as resp:
            if resp.status_code >= 400:
                return None
            ctype = resp.headers.get("content-type", "").lower()
            if "html" not in ctype and "text/plain" not in ctype:
                return None  # PDFs, images, ... keep the snippet
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                buf += chunk
                if len(buf) >= max_bytes:  # don't download huge pages
                    break
            encoding = resp.charset_encoding or "utf-8"
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return None

    try:
        body = buf.decode(encoding, errors="replace")
    except LookupError:  # unknown charset name in the header
        body = buf.decode("utf-8", errors="replace")
    text = body if "text/plain" in ctype else html_to_text(body)[1]
    text = text.strip()
    return text[:max_chars] if len(text) >= min_chars else None
