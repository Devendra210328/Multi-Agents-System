"""Binds the agents and chains into one workflow.

    topic -> search agent -> reader agent -> writer chain -> critic chain

Each stage reads the previous stage's slot in `state`, writes its own, and prints
what it produced before the next stage starts.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_core.messages import ToolMessage

from agent import build_critic_chain, build_reader_agent, build_search_agent, build_writer_chain
from config import MAX_URLS, SOURCE_CHARS

URL_PATTERN = re.compile(r"https?://\S+")


def as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def parse_payload(content: Any) -> Any:
    """Tool output is stringified into the message; recover the original value."""
    if not isinstance(content, str):
        return content
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except Exception:
            continue
    return content


def tool_payloads(result: dict[str, Any], tool_name: str) -> list[Any]:
    return [
        parse_payload(message.content)
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage) and message.name == tool_name
    ]


def final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    return as_text(messages[-1].content) if messages else ""


def format_sources(scraped: list[dict[str, Any]]) -> str:
    blocks = []
    for index, page in enumerate(scraped, start=1):
        content = (page.get("content") or "").strip()
        if content:
            blocks.append(
                f"[{index}] {page.get('title') or 'Untitled'}\n"
                f"URL: {page.get('url', '')}\n"
                f"{content[:SOURCE_CHARS]}"
            )
    return "\n\n---\n\n".join(blocks)


def banner(step: int, title: str) -> None:
    print(f"\n{'=' * 70}\n  STEP {step} — {title}\n{'=' * 70}", flush=True)


class ResearchPipeline:
    def __init__(self, max_urls: int = MAX_URLS, verbose: bool = True):
        self.max_urls = max_urls
        self.verbose = verbose
        self.search_agent = build_search_agent()
        self.reader_agent = build_reader_agent()
        self.writer_chain = build_writer_chain()
        self.critic_chain = build_critic_chain()

    def search(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.verbose:
            banner(1, "SEARCH AGENT")
            print(f"topic: {state['topic']}\n", flush=True)

        result = self.search_agent.invoke(
            {"messages": [{"role": "user", "content": state["topic"]}]}
        )

        seen: set[str] = set()
        hits: list[dict[str, str]] = []
        for batch in tool_payloads(result, "web_search_tool"):
            for hit in batch or []:
                url = hit.get("url", "") if isinstance(hit, dict) else ""
                if url and url not in seen:
                    seen.add(url)
                    hits.append(hit)

        state["search_result"] = hits
        state["search_notes"] = final_text(result)

        # Prefer the urls the agent shortlisted, but only ones the tool actually returned.
        shortlisted = [
            url.rstrip(".,);]")
            for url in URL_PATTERN.findall(state["search_notes"])
            if url.rstrip(".,);]") in seen
        ]
        ordered = list(dict.fromkeys(shortlisted)) or [hit["url"] for hit in hits]
        state["urls"] = ordered[: self.max_urls]

        if self.verbose:
            print(f"{len(hits)} results found:", flush=True)
            for hit in hits:
                print(f"  · {hit['url']}", flush=True)
                print(f"    {hit['content'][:120].strip()}...", flush=True)
            print(f"\nselected {len(state['urls'])} to read:", flush=True)
            for url in state["urls"]:
                print(f"  -> {url}", flush=True)

        return state

    def read(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.verbose:
            banner(2, "READER AGENT")

        listing = "\n".join(state["urls"])
        result = self.reader_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Research topic: {state['topic']}\n\n"
                            f"Scrape each of these urls and report what you found:\n{listing}"
                        ),
                    }
                ]
            }
        )

        scraped = [p for p in tool_payloads(result, "scrape_url_tool") if isinstance(p, dict)]
        state["scraped_content"] = scraped
        state["reading_notes"] = final_text(result)

        if self.verbose:
            for page in scraped:
                if page.get("content"):
                    print(
                        f"  [ok]   {page['word_count']:>5} words  "
                        f"{(page.get('title') or 'Untitled')[:55]}",
                        flush=True,
                    )
                    print(f"         {page['url']}  ({page['extracted_with']})", flush=True)
                else:
                    print(f"  [fail] {page.get('url', '')}", flush=True)
                    print(f"         {page.get('error', 'no content')}", flush=True)

        return state

    def write(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.verbose:
            banner(3, "WRITER CHAIN")

        state["sources"] = format_sources(state["scraped_content"])
        if not state["sources"]:
            raise RuntimeError("Every page failed to scrape — nothing to write from.")

        state["draft"] = self.writer_chain.invoke(
            {"topic": state["topic"], "sources": state["sources"]}
        )

        if self.verbose:
            print(state["draft"], flush=True)
            print(f"\n({len(state['draft'].split())} words)", flush=True)

        return state

    def critique(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.verbose:
            banner(4, "CRITIC CHAIN")

        critique = self.critic_chain.invoke(
            {"topic": state["topic"], "draft": state["draft"], "sources": state["sources"]}
        )
        state["critique"] = critique

        if self.verbose:
            print(f"score: {critique.score}/10\n", flush=True)
            print("strengths:", flush=True)
            for item in critique.strengths:
                print(f"  + {item}", flush=True)
            print("\nimprovements:", flush=True)
            for item in critique.improvements:
                print(f"  - {item}", flush=True)
            print(f"\nverdict: {critique.verdict}", flush=True)

        return state

    def run(self, topic: str) -> dict[str, Any]:
        state: dict[str, Any] = {"topic": topic}
        self.search(state)
        self.read(state)
        self.write(state)
        self.critique(state)
        return state


def run_pipeline(topic: str, max_urls: int = MAX_URLS, verbose: bool = True) -> dict[str, Any]:
    return ResearchPipeline(max_urls=max_urls, verbose=verbose).run(topic)
