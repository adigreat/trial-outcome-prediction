@HANDOFF.md

Rules for Claude in this repo:
- HANDOFF.md above is the source of truth. Read section 13 first: it says what is done and what is next for each person.
- Record decisions in HANDOFF.md section 9, and update sections 9, 10 and 13 before ending a session. The other person's Claude only knows what is written there.
- Ask which person the user is (A = Ananya, features; B = Aditya, modelling) and stay in that person's folders (`features/` = A, `eval/` and `models/` = B) unless told otherwise.
- Never use the `status` or `why_stop` columns as features. They leak the outcome.
- Never commit data, `.npy` files, or secrets. The repo is public.
- Never push to `main` or force-push. Work on a branch, open a PR, and ask before pushing, opening, or merging.
- Use the project venv: `.venv/Scripts/python` (create it from `requirements.txt`). Run checks from the repo root:
  `python -m eval.test_harness` and `python -m models.test_baselines`.
- With each new prompt, check for open PRs, review them, and ask for approval or merging.
- Keep `HANDOFF.md`, `EASY_LANGUAGE.md` and `README.md` in step. Whenever the project state changes (a PR merges, results come in, a decision is made, next steps change, a bug is found), update all three in the same change, before you finish the task. A stale doc counts as a bug. `HANDOFF.md` wins if they disagree.
