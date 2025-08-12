from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State
from prompts import SYSTEM_PROMPT, USER_PROMPT_SUBSEQUENT


THINK_GUESS_SYSTEM_PROMPT = SYSTEM_PROMPT


def parse_feedback_line(feedback: str) -> Tuple[str, str]:
    """Parse last env feedback line like "CRANE -> G- -Y - -" into (guess, pattern).
    Pattern chars: G=green, Y=yellow, -=grey. We tolerate spaces.
    """
    # Supports either:
    #   "CRANE -> G-Y--" OR
    #   "Guess 1: CRANE -> FEEDBACK: C(x) R(-) A(x) N(-) E(x)"
    s = feedback.strip()
    if "->" not in s:
        return "", ""
    left, right = s.split("->", 1)
    guess = left.split(":")[-1].strip().split()[0].strip().lower()
    # Try compact G/Y/- form
    compact = re.sub(r"[^GY-]", "", right.upper())
    if len(compact) == 5:
        return guess, compact
    # Try token form like B(✓) R(-) ... -> map to G/Y/-
    token_pattern = []
    for token in right.split():
        m = re.search(r"\((.)\)", token)
        if not m:
            continue
        sym = m.group(1)
        if sym == "✓":
            token_pattern.append("G")
        elif sym == "-":
            token_pattern.append("Y")
        elif sym in {"x", "X"}:
            token_pattern.append("-")
    pattern = "".join(token_pattern)
    return guess, pattern if len(pattern) == 5 else (guess, "")


def build_constraints_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute letter constraints from past (guess, pattern) pairs.
    Returns dict with:
      - fixed_positions: dict[pos]->char
      - excluded_positions: dict[pos]->set(chars)
      - required_letters: multiset as dict[char]->min_count
      - excluded_letters: set(chars that cannot appear at all)
    """
    fixed_positions: Dict[int, str] = {}
    excluded_positions: Dict[int, set[str]] = {}
    required_letters: Dict[str, int] = {}
    excluded_letters: set[str] = set()

    # Track min occurrences by counting Y/G per letter across all turns minus fixed positions
    for turn in history:
        guess = turn.get("guess", "").lower()
        fb = turn.get("feedback", "")
        # Support either Wordle-letter-coded string or compact code; accept both forms
        # Expected forms examples:
        #   "CRANE -> G-Y--" or "c r a n e\nGY- - -" etc. We'll try simple parser first.
        if turn.get("pattern"):
            pattern = turn["pattern"].upper()
        elif "->" in fb:
            _, pattern = parse_feedback_line(fb)
        else:
            # normalize tokens G/Y/- only
            pattern = re.sub(r"[^GY-]", "", fb.upper())
        if len(guess) != 5 or len(pattern) != 5:
            continue

        # per-letter accounting to avoid marking a letter fully excluded if it had Y/G
        letter_has_positive: Dict[str, bool] = {}
        for i, (gch, pch) in enumerate(zip(guess, pattern)):
            if pch == "G":
                fixed_positions[i] = gch
                letter_has_positive[gch] = True
                required_letters[gch] = max(required_letters.get(gch, 0), 1)
            elif pch == "Y":
                excluded_positions.setdefault(i, set()).add(gch)
                letter_has_positive[gch] = True
                required_letters[gch] = max(required_letters.get(gch, 0), 1)
            elif pch == "-":
                excluded_positions.setdefault(i, set()).add(gch)
        for ch, pos in letter_has_positive.items():
            pass
        # Letters that were only '-' across all occurrences in this turn and not positive elsewhere may be excluded
        for i, (gch, pch) in enumerate(zip(guess, pattern)):
            if pch == "-" and not letter_has_positive.get(gch, False):
                excluded_letters.add(gch)

    return {
        "fixed_positions": fixed_positions,
        "excluded_positions": excluded_positions,
        "required_letters": required_letters,
        "excluded_letters": excluded_letters,
    }


def check_word_against_constraints(word: str, constraints: Dict[str, Any]) -> bool:
    word = word.lower()
    if len(word) != 5 or not word.isalpha():
        return False
    fixed = constraints["fixed_positions"]
    ex_pos = constraints["excluded_positions"]
    req = constraints["required_letters"]
    ex_letters = constraints["excluded_letters"]

    for pos, ch in fixed.items():
        if word[pos] != ch:
            return False
    for pos, letters in ex_pos.items():
        if word[pos] in letters:
            return False
    for ch, min_count in req.items():
        if word.count(ch) < min_count:
            return False
    for ch in ex_letters:
        if ch in word:
            return False
    return True


def _get_last_assistant_content(completion: Messages) -> str:
    if not isinstance(completion, list):
        return ""
    for msg in reversed(completion):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _extract_guess_from_completion(parser: vf.Parser, completion: Messages) -> str | None:
    # Legacy fallback (kept for compatibility if called elsewhere)
    guess = parser.parse_answer(completion)
    if not guess:
        return None
    guess = guess.strip()
    if guess.startswith("[") and guess.endswith("]"):
        guess = guess[1:-1]
    return guess.lower()


def _sft_feedback_tokens(last_guess: str, answer: str) -> Tuple[str, str]:
    """Safe feedback tokenization: tolerate malformed lengths.

    Returns (tokens_str, pattern_str) with total length 5.
    """
    tokens: List[str] = []
    pattern_chars: List[str] = []
    g = (last_guess or "")[:5].lower()
    a = (answer or "")[:5].lower()
    L = min(len(g), len(a), 5)
    for i in range(L):
        ch = g[i]
        if i < len(a) and ch == a[i]:
            tokens.append(f"{ch.upper()}(✓)")
            pattern_chars.append("G")
        elif ch in a:
            tokens.append(f"{ch.upper()}(-)")
            pattern_chars.append("Y")
        else:
            tokens.append(f"{ch.upper()}(x)")
            pattern_chars.append("-")
    # pad to 5 if needed (shouldn't happen with valid data)
    while len(tokens) < 5:
        tokens.append("?(x)")
        pattern_chars.append("-")
    return " ".join(tokens), "".join(pattern_chars)


class WordleMultiTurnEnv(MultiTurnEnv):
    def __init__(
        self,
        valid_words: List[str],
        dataset=None,
        eval_dataset=None,
        system_prompt: str = THINK_GUESS_SYSTEM_PROMPT,
        max_turns: int = 6,
        expect_think: bool | None = None,
        **kwargs,
    ):
        parser = vf.XMLParser(fields=["think", "guess"], answer_field="guess")
        rubric = vf.Rubric(parser=parser)

        # Rewards: success within <=6 turns (or early-finish bonus), STRICT format adherence,
        # constraint validity, and STRICT length adherence of guesses.
        rubric.add_reward_func(self.correct_answer_reward)
        rubric.add_reward_func(self.early_finish_reward)
        # Replace library's permissive format reward with a strict variant
        rubric.add_reward_func(self.format_strict_reward, weight=0.2)
        rubric.add_reward_func(self.constraint_validity_reward, weight=0.5)
        rubric.add_reward_func(self.length_adherence_reward, weight=0.1)

        super().__init__(
            message_type="chat",
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
        self.valid_words = set(w.lower() for w in valid_words if len(w) == 5)
        # Whether a <think> block is expected (strict check). Defaults inferred from system prompt.
        sys_lower = (system_prompt or "").lower()
        self.expect_think: bool = (
            expect_think if expect_think is not None else ("<think>" in sys_lower)
        )

    # ---------- Strict formatting helpers ----------
    def _strict_extract_guess(self, completion: Messages) -> str | None:
        """Strictly extract guess only if the entire assistant content matches the expected format.

        If `self.expect_think` is True, require:
          ^\s*<think>...</think>\s*<guess>[ABCDE]</guess>\s*$
        Else, require:
          ^\s*<guess>[ABCDE]</guess>\s*$
        """
        content = _get_last_assistant_content(completion)
        if not content:
            return None
        if self.expect_think:
            m = re.match(
                r"^\s*<\s*think\s*>([\s\S]*?)<\s*/\s*think\s*>\s*<\s*guess\s*>\s*([A-Za-z]{5})\s*<\s*/\s*guess\s*>\s*$",
                content,
                flags=re.DOTALL,
            )
            if not m:
                return None
            guess = m.group(2)
        else:
            m = re.match(
                r"^\s*<\s*guess\s*>\s*([A-Za-z]{5})\s*<\s*/\s*guess\s*>\s*$",
                content,
            )
            if not m:
                return None
            guess = m.group(1)
        return guess.lower()

    def _has_valid_strict_format(self, completion: Messages) -> bool:
        return self._strict_extract_guess(completion) is not None

    # ---------- Sanitization for non-increasing templates ----------
    @staticmethod
    def _extract_think_and_guess(raw: str) -> tuple[str, str | None]:
        """Extract think text (without tags) and guess (inside <guess>).</n+        If no think, returns (raw_without_guess, guess).
        """
        # Extract guess
        m_guess = re.search(r"<\s*guess\s*>(.*?)<\s*/\s*guess\s*>", raw, re.DOTALL | re.IGNORECASE)
        guess = m_guess.group(1).strip() if m_guess else None
        # Extract think block
        m_think = re.search(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", raw, re.DOTALL | re.IGNORECASE)
        think = m_think.group(1).strip() if m_think else ""
        # Include any text between </think> and <guess>
        m_between = re.search(r"</\s*think\s*>(.*?)<\s*guess\s*>", raw, re.DOTALL | re.IGNORECASE)
        additional = m_between.group(1).strip() if m_between else ""
        if additional:
            think = (think + "\n\n" + additional).strip()
        return think, guess

    @classmethod
    def _sanitize_assistant_content(cls, content: str) -> str:
        """Remove <think> tags while preserving content and keep <guess> block, matching SFT framing.
        This stabilizes token prefixes for models like Qwen3 whose chat templates drop <think> at re-encode time.
        """
        think, guess = cls._extract_think_and_guess(content)
        if guess is None:
            # No guess block; return original (model may have malformed output)
            return content
        return (think + ("\n\n" if think else "") + f"<guess>{guess}</guess>").strip()

    @classmethod
    def _sanitize_completion_chat(cls, completion: Messages) -> Messages:
        if not isinstance(completion, list):
            return completion
        sanitized: list[dict] = []
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                new_msg = dict(msg)
                new_msg["content"] = cls._sanitize_assistant_content(msg["content"])
                sanitized.append(new_msg)
            else:
                sanitized.append(msg)
        return sanitized

    # Override to sanitize assistant messages prior to token alignment with vLLM outputs
    def process_env_results_vllm(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        states: list[State],
        rewards: list[float],
        processing_class,
        max_seq_len: int = -1,
        mask_env_responses: bool = False,
        mask_truncated_completions: bool = False,
        zero_truncated_completions: bool = False,
    ):
        completions_sanitized = [self._sanitize_completion_chat(c) for c in completions]
        return super().process_env_results_vllm(
            prompts=prompts,
            completions=completions_sanitized,
            states=states,
            rewards=rewards,
            processing_class=processing_class,
            max_seq_len=max_seq_len,
            mask_env_responses=mask_env_responses,
            mask_truncated_completions=mask_truncated_completions,
            zero_truncated_completions=zero_truncated_completions,
        )

    # Safer variant that tolerates minor template differences by using longest-common-prefix instead of strict assert
    def process_chat_format_vllm(
        self,
        prompt: list[dict],
        completion: list[dict],
        state: State,
        processing_class,
        mask_env_responses: bool = False,
    ):
        responses = state["responses"]
        responses_idx = 0
        zipped = []
        for turn in completion:
            if turn["role"] == "assistant":
                zipped.append((turn, responses[responses_idx]))
                responses_idx += 1
            else:
                zipped.append((turn, None))
        assert len(responses) == responses_idx, "Responses not fully consumed"
        assert len(zipped) == len(completion), "Length mismatch"
        prompt_ids: list[int] = processing_class.apply_chat_template(
            conversation=prompt,  # type: ignore
            add_generation_prompt=True,
        )
        messages_consumed = prompt.copy()
        prompt_mask: list[int] = [0] * len(prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[int] = []
        completion_logprobs: list[float] = []
        i = 0
        while i < len(zipped):
            message, response = zipped[i]
            if message["role"] == "assistant":
                assert response is not None, "Response should not be None"
                # parse generated tokens/logprobs from vLLM
                tokens = [
                    int(t.token.split(":")[-1])
                    for t in response.choices[0].logprobs.content
                ]
                logs = [lp.logprob for lp in response.choices[0].logprobs.content]
                completion_ids.extend(tokens)
                completion_mask.extend([1] * len(tokens))
                completion_logprobs.extend(logs)
                messages_consumed.append(message)
                i += 1
            else:
                # accumulate consecutive non-assistant messages
                consecutive_messages = [message]
                j = i + 1
                while j < len(zipped) and zipped[j][0]["role"] != "assistant":
                    consecutive_messages.append(zipped[j][0])
                    j += 1
                prefix = processing_class.apply_chat_template(
                    conversation=messages_consumed
                )
                with_turn = processing_class.apply_chat_template(
                    conversation=messages_consumed + consecutive_messages
                )
                # longest common prefix
                k = 0
                kmax = min(len(prefix), len(with_turn))
                while k < kmax and prefix[k] == with_turn[k]:
                    k += 1
                new_ids = with_turn[k:]
                if mask_env_responses:
                    new_mask = [0] * len(new_ids)
                else:
                    new_mask = [1] * len(new_ids)
                completion_ids.extend(new_ids)
                completion_mask.extend(new_mask)
                completion_logprobs.extend([0.0] * len(new_ids))
                messages_consumed.extend(consecutive_messages)
                i = j
        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        )

    # ---------- Rewards ----------
    def correct_answer_reward(self, parser: vf.Parser, completion: Messages, answer: str, **kwargs) -> float:
        guess = self._strict_extract_guess(completion)
        return 1.0 if guess and guess == answer else 0.0

    def early_finish_reward(self, parser: vf.Parser, completion: Messages, answer: str, **kwargs) -> float:
        # 1/(turns) if success; else 0
        assistants = [m for m in completion if isinstance(m, dict) and m.get("role") == "assistant"]
        guess = self._strict_extract_guess(completion)
        if guess and guess == answer and assistants:
            return 1.0 / max(1, len(assistants))
        return 0.0

    def constraint_validity_reward(self, parser: vf.Parser, completion: Messages, state: State, **kwargs) -> float:
        guess = self._strict_extract_guess(completion)
        if not guess or guess not in self.valid_words:
            return 0.0
        constraints = build_constraints_from_history(state.get("history", []))
        return 1.0 if check_word_against_constraints(guess, constraints) else 0.0

    def length_adherence_reward(self, parser: vf.Parser, completion: Messages, **kwargs) -> float:
        """Give reward only if the entire assistant content strictly matches the expected tag format
        and the <guess> contains exactly 5 letters in brackets.
        """
        guess = self._strict_extract_guess(completion)
        return 1.0 if guess and len(guess) == 5 and guess.isalpha() else 0.0

    def format_strict_reward(self, parser: vf.Parser, completion: Messages, **kwargs) -> float:
        """Binary reward: 1.0 only if the assistant output strictly matches the required XML format.

        - With think: '<think>...</think>\n<guess>[APPLE]</guess>' and nothing else
        - Without think: '<guess>[APPLE]</guess>' and nothing else
        """
        return 1.0 if self._has_valid_strict_format(completion) else 0.0

    # ---------- MultiTurn protocol ----------
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        # Complete if: last assistant guess equals the answer OR reached max_turns
        guess = self._strict_extract_guess(messages)
        if guess and state.get("answer") and guess == state["answer"]:
            return True
        assistants = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        return len(assistants) >= self.max_turns

    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        # Get last assistant guess
        guess = self._strict_extract_guess(messages)
        if not guess:
            # Ask for a guess in the required format
            response = [
                {
                    "role": "user",
                    "content": (
                        "Your output was invalid. Respond in EXACT XML format only.\n"
                        + (
                            "<think>your reasoning</think>\n<guess>apple</guess>"
                            if self.expect_think
                            else "<guess>apple</guess>"
                        )
                        + "\nNo extra text outside these tags."
                    ),
                }
            ]
            return response, state

        # Build feedback relative to answer in SFT style
        answer: str = state.get("answer", "")
        # Guard against malformed answers to avoid crashing the run
        if len(answer) != 5 or not answer.isalpha():
            state["failed"] = True
            return [
                {
                    "role": "user",
                    "content": "Invalid puzzle state (bad answer). Skipping this example.",
                }
            ], state
        tokens_str, pattern = _sft_feedback_tokens(guess, answer)
        turn_num = state.get("turn", 0)
        feedback = f"Guess {turn_num}: {guess} -> FEEDBACK: {tokens_str}"

        # Append structured history used for constraint checking
        hist = state.get("history", [])
        hist.append({"guess": guess, "feedback": feedback, "pattern": pattern})
        state["history"] = hist

        # Maintain cumulative feedback string like SFT create_sft_data
        cum = state.get("feedback_history_str", "")
        cum = (cum + "\n" if cum else "") + feedback
        state["feedback_history_str"] = cum

        # If solved, end; else, send feedback
        if guess == answer:
            return [
                {
                    "role": "user",
                    "content": f"Correct. Game finished in {len(hist)} turns.",
                }
            ], state
        # Frame next user message as in SFT
        return [
            {
                "role": "user",
                "content": USER_PROMPT_SUBSEQUENT.format(
                    feedback_str=state["feedback_history_str"]
                ),
            }
        ], state


def load_environment(
    valid_words_path: str,
    dataset: Any = None,
    eval_dataset: Any = None,
    use_think: bool = True,
    max_turns: int = 6,
):
    valid_words = [w.strip() for w in open(valid_words_path, "r").read().splitlines() if w.strip()]
    return WordleMultiTurnEnv(
        valid_words=valid_words,
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=THINK_GUESS_SYSTEM_PROMPT if use_think else "Respond only with <guess>[word]</guess>",
        max_turns=max_turns,
        expect_think=use_think,
    )