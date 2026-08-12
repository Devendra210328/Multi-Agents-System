"""Entry point.

Usage:
    python main.py "impact of RAG on enterprise search"
"""

from __future__ import annotations

import sys

from pipeline import run_pipeline

# Scraped page text contains characters cp1252 cannot represent.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        topic = "What is retrieval-augmented generation, and where does it fall short?"

    state = run_pipeline(topic)
    print(f"\n{'=' * 70}\n  DONE — {len(state['scraped_content'])} sources read\n{'=' * 70}")


if __name__ == "__main__":
    main()
