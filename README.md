# Tetris GA arena

Evaluates agents submitted to [Tetris GA](https://huggingface.co/spaces/GonzaloFuentes1/tetris-ga) and keeps the [leaderboard](https://huggingface.co/datasets/GonzaloFuentes1/tetris-ga-leaderboard) up to date.

How a submission works:

1. In the Space you sign in with Hugging Face and press *Submit to leaderboard*. The page opens a discussion titled `[submission] <name>` on the leaderboard dataset, with the agent as JSON. Hugging Face records who opened it.
2. Every 15 minutes, [a GitHub Action](.github/workflows/evaluate.yml) runs [`evaluate.js`](evaluate.js). It plays each new agent on the four levels with secret seeds, so scores cannot be faked, rates it against the 100 bots on the same games and keeps each user's best entry in `leaderboard.json`.
3. It replies in the discussion with the scores and closes it.

Agents with custom rules are not accepted, because that would mean running other people's code here.

Secrets: `HF_TOKEN` (a fine-grained Hugging Face token with write access to the leaderboard dataset and its discussions) and `EVAL_SEEDS`.
