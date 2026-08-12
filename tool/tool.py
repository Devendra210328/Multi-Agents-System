"""Tools the agents call: live web search and page extraction."""

from __future__ import annotations

import os
from typing import Any

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.tools import tool

from config import (
    MAX_CONTENT_CHARS,
    MAX_RESULTS,
    REQUEST_TIMEOUT,
    SNIPPET_CHARS,
    USER_AGENT,
)


def _tavily_client():
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in .env")
    return TavilyClient(api_key=api_key)


@tool
def web_search_tool(query: str) -> list[dict[str, str]]:
    """Search the live web and return the top results.

    Each result is {"url", "content"} where content is a short snippet. Pass a
    promising url to scrape_url_tool to read the full page.
    """
    try:
        response = _tavily_client().search(
            query=query, max_results=MAX_RESULTS, search_depth="advanced"
        )
    except Exception as exc:
        return [{"url": "", "content": f"Search failed: {exc}"}]

    return [
        {"url": hit.get("url", ""), "content": (hit.get("content") or "")[:SNIPPET_CHARS]}
        for hit in response.get("results", [])
    ]


def _extract_with_trafilatura(url: str) -> dict[str, Any] | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    text = trafilatura.extract(
        downloaded, include_comments=False, include_tables=True, favor_precision=True
    )
    if not text:
        return None

    meta = trafilatura.extract_metadata(downloaded)
    return {
        "title": (getattr(meta, "title", None) or "").strip(),
        "author": (getattr(meta, "author", None) or "").strip(),
        "published": (getattr(meta, "date", None) or "").strip(),
        "content": text.strip(),
        "extracted_with": "trafilatura",
    }


def _extract_with_soup(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for junk in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        junk.decompose()

    lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
    return {
        "title": (soup.title.string or "").strip() if soup.title else "",
        "author": "",
        "published": "",
        "content": "\n".join(line for line in lines if line),
        "extracted_with": "beautifulsoup",
    }


@tool
def scrape_url_tool(url: str) -> dict[str, Any]:
    """Fetch a web page and return its main article text in structured form.

    Returns {"url", "title", "author", "published", "content", "word_count",
    "extracted_with"}. On failure the same shape comes back with an "error" field
    and empty content — try a different url.
    """
    try:
        record = _extract_with_trafilatura(url) or _extract_with_soup(url)
    except Exception as exc:
        return {
            "url": url,
            "title": "",
            "author": "",
            "published": "",
            "content": "",
            "word_count": 0,
            "extracted_with": "none",
            "error": f"{type(exc).__name__}: {exc}",
        }

    content = record["content"][:MAX_CONTENT_CHARS]
    return {
        "url": url,
        "title": record["title"],
        "author": record["author"],
        "published": record["published"],
        "content": content,
        "word_count": len(content.split()),
        "extracted_with": record["extracted_with"],
    }


def to_documents(scraped: list[dict[str, Any]]) -> list[Document]:
    """Convert scrape results into LangChain Documents."""
    return [
        Document(
            page_content=page["content"],
            metadata={
                "source": page.get("url", ""),
                "title": page.get("title", ""),
                "author": page.get("author", ""),
                "published": page.get("published", ""),
                "extracted_with": page.get("extracted_with", ""),
            },
        )
        for page in scraped
        if page.get("content")
    ]
