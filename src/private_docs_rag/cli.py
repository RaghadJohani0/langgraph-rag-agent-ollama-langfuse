"""Command line entry point.

Usage:
    python -m private_docs_rag.cli "What is RAG?"
"""

import sys

from private_docs_rag.llm.client import LLMClient, LLMError

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer briefly. "
    "If you are not sure, say you don't know instead of guessing."
)


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m private_docs_rag.cli "your question"')
        return 1

    question = " ".join(sys.argv[1:])
    client = LLMClient()
    try:
        print(client.ask(question, system=SYSTEM_PROMPT))
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
