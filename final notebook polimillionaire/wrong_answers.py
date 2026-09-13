"""Wrong answer logger.

Appends one JSON record per incorrect or timed-out answer to wrong_answers.jsonl.
Each record contains everything needed for post-game analysis:
username, competition, level, question, options, reasoning, answer given, RAG context,
hardware resources, and whether the question timed out.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

WRONG_ANSWERS_FILE = Path("wrong_answers.jsonl")


def save_wrong_answer(
    username: str,
    competition_id: int,
    level: int,
    question: str,
    options: dict,
    reasoning: str,
    answer_given: str,
    rag_context: str | None = None,
    resources: dict | None = None,
    timed_out: bool = False,
) -> None:
    """Append a wrong or timed-out answer record to wrong_answers.jsonl."""
    record = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "username":       username,
        "competition_id": competition_id,
        "level":          level,
        "question":       question,
        "options":        {str(k): v for k, v in options.items()},
        "reasoning":      reasoning,
        "answer_given":   answer_given,
        "rag_context":    rag_context,
        "resources":      resources,
        "timed_out":      timed_out,
    }
    with open(WRONG_ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    label = "Timeout" if timed_out else "Wrong answer"
    logging.info("%s saved to %s (level %d)", label, WRONG_ANSWERS_FILE, level)
