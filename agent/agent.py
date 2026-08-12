"""The two autonomous agents. This module builds them; the pipeline runs them."""

from __future__ import annotations

from langchain.agents import create_agent

from config import get_model
from tool import scrape_url_tool, web_search_tool

SEARCH_SYSTEM_PROMPT = """You are the search agent in a research pipeline.

Given a research topic, call web_search_tool to find current sources. Run more than
one search when the topic has distinct sub-questions — vary the wording rather than
repeating the same query.

Each result gives you a url and a short snippet. The snippet is only a preview; you
are not answering the question here. Judge from it which sources look substantive and
on-topic, and skip duplicates, listicles, and pure marketing pages.

Finish by listing the urls worth reading in full, one per line, best first. No
commentary around the list."""

READER_SYSTEM_PROMPT = """You are the reader agent in a research pipeline.

You are given urls the search agent selected. Call scrape_url_tool on each one to
pull its full text. Scrape every url you are given before you answer.

Some pages fail — the tool returns an `error` field and empty content. That is
expected: note the failure, move on to the next url, and never retry the same url
more than once.

Finish with a short inventory of what you gathered: for each url, its title and one
line on what it covers. Do not summarize the topic itself — the writer chain does
that from the full text."""


def build_search_agent(model=None):
    """Agent that finds sources. Tool: web_search_tool."""
    return create_agent(
        model or get_model(), [web_search_tool], system_prompt=SEARCH_SYSTEM_PROMPT
    )


def build_reader_agent(model=None):
    """Agent that reads sources. Tool: scrape_url_tool."""
    return create_agent(
        model or get_model(), [scrape_url_tool], system_prompt=READER_SYSTEM_PROMPT
    )
