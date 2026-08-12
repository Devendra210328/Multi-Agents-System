"""The writer and critic chains. This module builds them; the pipeline runs them."""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import get_model

WRITER_SYSTEM_PROMPT = """You are the writer in a research pipeline.

Write a clear, well-organized report on the topic using only the source material you
are given. You are synthesizing, not searching: every claim must trace back to the
sources, and if they disagree, say so rather than picking a side silently.

Structure it with short headed sections and lead with what matters. If the sources do
not cover part of the topic, state that gap plainly instead of filling it from your
own knowledge.

Close with a Sources section listing the urls you actually drew on."""

WRITER_HUMAN_PROMPT = """Topic: {topic}

Source material:
{sources}"""

CRITIC_SYSTEM_PROMPT = """You are the critic in a research pipeline. You grade a draft
report against the source material it was written from.

Judge it on: factual support (is every claim traceable to the sources?), coverage of
the topic, structure and readability, and whether it names its sources.

Be specific and useful. "Add more detail" is worthless; "the section on cost cites no
source and contradicts the Databricks page" is what the writer can act on. Reserve 9
and 10 for drafts you cannot meaningfully improve.

{format_instructions}"""

CRITIC_HUMAN_PROMPT = """Topic: {topic}

Draft report:
{draft}

Source material the draft was written from:
{sources}"""


class Critique(BaseModel):
    """The critic's structured verdict on a draft."""

    score: int = Field(description="Overall quality from 1 to 10", ge=1, le=10)
    strengths: list[str] = Field(description="What the draft does well, one point each")
    improvements: list[str] = Field(description="Specific, actionable fixes, one point each")
    verdict: str = Field(description="A single-line summary judgement")


def build_writer_chain(model=None):
    """prompt | llm | StrOutputParser(). Takes {topic, sources}, returns report text."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", WRITER_SYSTEM_PROMPT), ("human", WRITER_HUMAN_PROMPT)]
    )
    return prompt | (model or get_model(temperature=0.3)) | StrOutputParser()


def build_critic_chain(model=None):
    """prompt | llm | PydanticOutputParser(). Takes {topic, draft, sources}."""
    parser = PydanticOutputParser(pydantic_object=Critique)
    prompt = ChatPromptTemplate.from_messages(
        [("system", CRITIC_SYSTEM_PROMPT), ("human", CRITIC_HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | (model or get_model(temperature=0)) | parser
