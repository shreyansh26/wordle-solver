SYSTEM_PROMPT = '''### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example:
Secret Word: BRISK

Guess 1: STORM -> Feedback: S(-) T(x) O(x) R(-) M(x)
Guess 2: BRAVE -> Feedback: B(✓) R(✓) A(x) V(x) E(x)
Guess 3: BRISK -> Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

### Response Format:
Think through the problem and feedback step by step. Make sure to first add your step-by-step thought process within <think> </think> tags. Then, return your guessed word in the following format: <guess> guessed-word </guess>.
'''

USER_PROMPT_FIRST = '''Make your first 5-letter word guess.'''

USER_PROMPT_SUBSEQUENT = '''Make a new 5-letter word guess.

Here is some previous feedback:
{feedback_str}
'''

ASSISTANT_PROMPT_FIRST = '''Let me solve this step by step.'''