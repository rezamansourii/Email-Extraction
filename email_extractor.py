#!/usr/bin/env python3
"""
Email extractor from web pages (single URLs or a crawl).

Features:
- Extracts standard emails (name@domain.tld)
- Tries to de-obfuscate common patterns: "name [at] domain [dot] com", "name (at) domain.com"
- Optional same-domain crawl with depth and page limits
- Skips mailto: links duplication naturally (regex catches them too)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from requests import Session
from requests.exceptions import ProxyError
from bs4 import BeautifulSoup, Comment

# Standard email pattern (kept conservative to reduce false positives)
EMAIL_RE = re.compile(
    r"""(?ix)
    \b
    [a-z0-9._%+\-]+
    @
    (?:[a-z0-9\-]+\.)+
    [a-z]{2,}
    \b
    """
)

# Obfuscated patterns -> we'll normalize page text before running EMAIL_RE
OBFUSCATIONS = [
    # [at], (at), {at}, " at "
    (re.compile(r"(?i)\s*(\[\s*at\s*\]|\(\s*at\s*\)|\{\s*at\s*\}|\s+at\s+)\s*"), "@"),
    # [dot], (dot), {dot}, " dot "
    (re.compile(r"(?i)\s*(\[\s*dot\s*\]|\(\s*dot\s*\)|\{\s*dot\s*\}|\s+dot\s+)\s*"), "."),
    # HTML entities & common obfuscation symbols
    (re.compile(r"(?i)\s*(?:&#64;|&commat;)\s*"), "@"),
    (re.compile(r"(?i)\s*(?:&#46;|&period;)\s*"), "."),
]

# Simple filters to reduce obvious junk matches
BAD_PREFIXES = {"example", "test"}
BAD_DOMAINS = {"example.com", "example.org", "example.net"}


@dataclass(frozen=True)
class CrawlConfig:
    depth: int = 0
    max_pages: int = 25
    delay_seconds: float = 0.0
    timeout_seconds: float = 15.0
    user_agent: str = "EmailExtractor/1.0 (+https://example.local)"


def normalize_text_for_emails(text: str) -> str:
    normalized = text
    for pattern, repl in OBFUSCATIONS:
        normalized = pattern.sub(repl, normalized)
    # Remove spaces around @ and dots that appear after normalization
    normalized = re.sub(r"\s*@\s*", "@", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    return normalized


def clean_emails(emails: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for e in emails:
        e = e.strip().strip(".,;:!?)\"]}'")
        e_lower = e.lower()

        # Basic sanity checks
        if len(e_lower) < 6 or len(e_lower) > 254:
            continue
        if e_lower.split("@", 1)[0] in BAD_PREFIXES:
            continue
        if e_lower.split("@", 1)[-1] in BAD_DOMAINS:
            continue
        out.add(e_lower)
    return out


def _get_response(session: Session, url: str, cfg: CrawlConfig) -> requests.Response:
    headers = {"User-Agent": cfg.user_agent}
    return session.get(url, headers=headers, timeout=cfg.timeout_seconds)


def fetch(url: str, cfg: CrawlConfig) -> Optional[str]:
    with requests.Session() as session:
        try:
            r = _get_response(session, url, cfg)
        except ProxyError:
            session.trust_env = False
            try:
                r = _get_response(session, url, cfg)
            except requests.RequestException as ex:
                print(f"[warn] fetch failed: {url} ({ex})", file=sys.stderr)
                return None
        except requests.RequestException as ex:
            print(f"[warn] fetch failed: {url} ({ex})", file=sys.stderr)
            return None

    try:
        r.raise_for_status()
    except requests.RequestException as ex:
        print(f"[warn] fetch failed: {url} ({ex})", file=sys.stderr)
        return None

    ctype = r.headers.get("Content-Type", "")
    # Skip obvious non-HTML responses
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype and not r.text.lstrip().startswith("<"):
        return None
    return r.text


def extract_emails_from_html(html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")

    # Get visible-ish text; also include attributes (mailto, data-email, etc.)
    text = soup.get_text(" ", strip=True)

    # Include some attribute content where emails often hide
    attrs_text_parts = []
    for tag in soup.find_all(True):
        for attr in ("href", "data-email", "data-mail", "content", "value", "data-user", "data-domain"):
            v = tag.get(attr)
            if isinstance(v, str) and v:
                attrs_text_parts.append(v)
        data_user = tag.get("data-user")
        data_domain = tag.get("data-domain")
        if isinstance(data_user, str) and isinstance(data_domain, str) and data_user and data_domain:
            attrs_text_parts.append(f"{data_user}@{data_domain}")

    comment_text = " ".join(comment.strip() for comment in soup.find_all(string=lambda node: isinstance(node, Comment)))

    combined = " ".join([text, comment_text] + attrs_text_parts)
    normalized = normalize_text_for_emails(combined)

    found = set(EMAIL_RE.findall(normalized))
    # EMAIL_RE with groups returns only the group if we used groups; we didn't, but be safe:
    if found and isinstance(next(iter(found)), tuple):
        found = {x[0] for x in found}  # type: ignore

    return clean_emails(found)


def same_domain(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.netloc.lower() == pb.netloc.lower() and pa.scheme in ("http", "https") and pb.scheme in (
        "http",
        "https",
    )


def extract_links(base_url: str, html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        abs_url, _frag = urldefrag(abs_url)  # drop #fragment
        p = urlparse(abs_url)
        if p.scheme in ("http", "https") and p.netloc:
            links.add(abs_url)
    return links


def crawl_and_extract(start_urls: Iterable[str], cfg: CrawlConfig, do_crawl: bool) -> Tuple[Set[str], Set[str]]:
    emails: Set[str] = set()
    visited: Set[str] = set()

    q = deque()
    normalized_start_urls = []
    for u in start_urls:
        normalized, _frag = urldefrag(u)
        normalized_start_urls.append(normalized)
        q.append((normalized, 0))

    start_domains = {urlparse(u).netloc.lower() for u in normalized_start_urls}

    while q and len(visited) < cfg.max_pages:
        url, depth = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        html = fetch(url, cfg)
        if html is None:
            continue

        emails |= extract_emails_from_html(html)

        if do_crawl and depth < cfg.depth:
            for link in extract_links(url, html):
                if urlparse(link).netloc.lower() in start_domains and link not in visited:
                    q.append((link, depth + 1))

        if cfg.delay_seconds > 0:
            time.sleep(cfg.delay_seconds)

    return emails, visited


def read_urls_from_file(path: str) -> list[str]:
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            urls.append(u)
    return urls


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract email addresses from web pages.")
    ap.add_argument("urls", nargs="*", help="One or more URLs.")
    ap.add_argument("--input", "-i", help="Path to a text file with URLs (one per line).")
    ap.add_argument("--output", "-o", help="Write found emails to this file (one per line).")
    ap.add_argument("--crawl", action="store_true", help="Enable same-domain crawling from the start URLs.")
    ap.add_argument("--depth", type=int, default=1, help="Crawl depth (requires --crawl). Default: 1")
    ap.add_argument("--max-pages", type=int, default=25, help="Maximum pages to fetch total. Default: 25")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between requests (seconds). Default: 0")
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout (seconds). Default: 15")
    ap.add_argument("--user-agent", default="EmailExtractor/1.0 (+https://example.local)", help="User-Agent header.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    start_urls = list(args.urls)
    if args.input:
        start_urls += read_urls_from_file(args.input)

    if not start_urls:
        ap.error("Provide at least one URL or use --input.")

    cfg = CrawlConfig(
        depth=max(0, args.depth),
        max_pages=max(1, args.max_pages),
        delay_seconds=max(0.0, args.delay),
        timeout_seconds=max(1.0, args.timeout),
        user_agent=args.user_agent,
    )

    emails, visited = crawl_and_extract(start_urls, cfg, do_crawl=args.crawl)

    for e in sorted(emails):
        print(e)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for e in sorted(emails):
                f.write(e + "\n")

    print(f"[info] visited {len(visited)} pages; found {len(emails)} unique emails", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
