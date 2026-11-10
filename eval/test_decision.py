"""Self-check on hand-made data. Run from the repo root: python -m eval.test_decision"""
from eval.decision import best_threshold, threshold_table

# 4 failures (0) with low scores, 4 successes (1) with high scores, one of each in the middle
y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
p = [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95, 0.60, 0.40]

rows = {r["threshold"]: r for r in threshold_table(y, p, saved_per_true_kill=10, lost_per_wrong_kill=30,
                                                   thresholds=[0.01, 0.3, 0.5, 0.7, 1.0])}
assert rows[0.01]["n_killed"] == 0 and rows[0.01]["net"] == 0 and rows[0.01]["kill_precision"] == ""
# below 0.3: the four failures only
assert (rows[0.3]["true_kills"], rows[0.3]["wrong_kills"], rows[0.3]["net"]) == (4, 0, 40)
# below 0.5: adds the success scored 0.40, so one wrong kill costs 30
assert (rows[0.5]["true_kills"], rows[0.5]["wrong_kills"], rows[0.5]["net"]) == (4, 1, 10)
# below 0.7: adds the failure scored 0.60
assert (rows[0.7]["true_kills"], rows[0.7]["wrong_kills"], rows[0.7]["net"]) == (5, 1, 20)
# threshold 1.0 kills everything: 5 failures, 5 successes
assert (rows[1.0]["n_killed"], rows[1.0]["net"]) == (10, 5 * 10 - 5 * 30)
assert rows[0.3]["failures_caught"] == 0.8 and rows[0.3]["successes_lost"] == 0.0

all_rows = threshold_table(y, p, 10, 30)
assert best_threshold(all_rows)["net"] == max(r["net"] for r in all_rows)
assert best_threshold(threshold_table(y, p, 10, 0))["net"] == 50  # free wrong kills: kill all 5 failures at best

for bad in (lambda: threshold_table([0, 2], [0.1, 0.2], 1, 1), lambda: threshold_table([0], [0.1, 0.2], 1, 1),
            lambda: threshold_table([0, 1], [0.1, 0.2], -1, 1)):
    try:
        bad()
    except AssertionError:
        continue
    raise SystemExit("expected an AssertionError")
print("ok")
