from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset
from prompts import SYSTEM_PROMPT, USER_PROMPT_FIRST


@dataclass
class WordleExample:
    # Minimal fields we rely on
    word: str  # ground-truth answer (5-letter)
    # History where each turn contains model guess and env feedback string
    # Expected JSONL: {"word": "APPLE", "turns": [{"guess": "ARISE", "feedback": "A- -- -P - -"}, ...]}
    turns: List[Dict[str, Any]]


def _normalize_word(x: str) -> str:
    return x.strip().lower()


def _format_system_prompt(use_think: bool = True) -> str:
    if use_think:
        return (
            "You are playing Wordle. Always respond ONLY in this XML format per turn:\n"
            "<think>your reasoning</think>\n<guess>[APPLE]</guess>\n"
            "The <guess> must contain a single 5-letter guess inside brackets."
        )
    return (
        "You are playing Wordle. Always respond ONLY with:\n<guess>[APPLE]</guess>"
    )


def _format_user_feedback(history_feedback: str) -> str:
    # We pass only the feedback text to the env; model will have full chat via env
    return f"Feedback: {history_feedback.strip()}"


def load_jsonl_as_dataset(
    jsonl_path: str | Path,
    use_think: bool = True,
    limit: int = -1,
) -> Dataset:
    jsonl_path = Path(jsonl_path)
    rows: List[Dict[str, Any]] = []
    sys_prompt = _format_system_prompt(use_think=use_think)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            data = json.loads(line)
            # Support both {prompt, answer} and {word, turns}
            if "prompt" in data and "answer" in data:
                prompt = data["prompt"]
                answer = _normalize_word(data["answer"])
                # Enforce valid 5-letter answers; skip invalid rows
                if len(answer) != 5 or not answer.isalpha():
                    continue
                info = data.get("info", {})
            else:
                answer = _normalize_word(data.get("word", ""))
                # Enforce valid 5-letter answers; skip invalid rows
                if len(answer) != 5 or not answer.isalpha():
                    continue
                turns = data.get("turns", [])
                # Build a chat-style initial prompt identical to SFT framing
                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_FIRST},
                ]
                info = {"initial_feedback": "", "history": turns}

            rows.append({"prompt": prompt, "answer": answer, "info": info})

    return Dataset.from_list(rows)



