"""Decision analysis (Person B): at what predicted success probability would a sponsor stop a trial?

Label 1 = success, so a trial is "killed" when its predicted success probability is below the
threshold. Costs are inputs, not facts: the sources are an open question (HANDOFF section 10).
"""
import numpy as np


def threshold_table(y_true, p_success, saved_per_true_kill, lost_per_wrong_kill, thresholds=None):
    """One row per threshold: who we would kill, who we wrongly kill, and the net value.

    saved_per_true_kill:  money saved by stopping a trial that would have failed (>= 0).
    lost_per_wrong_kill:  value lost by stopping a trial that would have succeeded (>= 0).
    net = true_kills * saved - wrong_kills * lost. Same currency for both.
    """
    y, p = np.asarray(y_true), np.asarray(p_success, dtype=float)
    assert len(y) == len(p) and len(y) > 0, "y_true and p_success must be the same non-zero length"
    assert set(np.unique(y)) <= {0, 1}, "labels must be 0/1"
    assert saved_per_true_kill >= 0 and lost_per_wrong_kill >= 0, "costs must be >= 0"
    ts = np.linspace(0.05, 0.95, 19) if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    for t in ts:
        kill = p < t
        true_kills = int((kill & (y == 0)).sum())
        wrong_kills = int((kill & (y == 1)).sum())
        n_kill = true_kills + wrong_kills
        rows.append({
            "threshold": round(float(t), 4), "n_killed": n_kill,
            "true_kills": true_kills, "wrong_kills": wrong_kills,
            "kill_precision": round(true_kills / n_kill, 4) if n_kill else "",
            "failures_caught": round(true_kills / max(int((y == 0).sum()), 1), 4),
            "successes_lost": round(wrong_kills / max(int((y == 1).sum()), 1), 4),
            "net": true_kills * saved_per_true_kill - wrong_kills * lost_per_wrong_kill,
        })
    return rows


def best_threshold(rows):
    """Row with the highest net value. Ties go to the lower threshold (kills fewer trials)."""
    return max(rows, key=lambda r: (r["net"], -r["threshold"]))
