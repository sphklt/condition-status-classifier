"""
Status trajectory tracking across sentences within a section.

Models how a condition's status evolves across multiple sentences
within the same section. The trajectory-reconciled status uses
time-decayed log-evidence accumulation: recent mentions dominate
but earlier signals still contribute.

Novel contribution over single-sentence classification:
  - resolution:  ongoing → resolved across sentences (correct final status)
  - relapse:     resolved → ongoing (flag for clinical review)
  - contradiction: negated → ongoing (flag for human review)
  - clarification: ambiguous → confident label (certainty gained)

Each transition type adds a bonus LLR to the destination label, encoding
the clinical prior that the most recent status is the most informative
while earlier statuses provide directional context.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable

LABELS = ("ongoing", "resolved", "negated", "ambiguous")

# Time-decay factor: point i from the end contributes _DECAY^(n-1-i).
# 0.7 → second-to-last = 70%, third-to-last = 49%, etc.
_DECAY = 0.7

# Bonus LLR added to the destination label when a clinically significant
# transition fires. Encodes the prior that these transitions carry strong
# directional evidence beyond the per-point confidence scores.
_TRANSITION_BONUS: dict[tuple[str, str], float] = {
    ("ongoing",   "resolved"):  0.80,
    ("ambiguous", "resolved"):  0.60,
    ("ambiguous", "ongoing"):   0.60,
    ("resolved",  "ongoing"):   1.00,  # relapse — strong recency signal
    ("negated",   "ongoing"):   0.90,  # contradiction
    ("negated",   "ambiguous"): 0.60,
    ("ongoing",   "ambiguous"): 0.40,
}

_TRANSITION_TYPE: dict[tuple[str, str], str] = {
    ("ongoing",   "resolved"):  "resolution",
    ("ambiguous", "resolved"):  "clarification_resolved",
    ("ambiguous", "ongoing"):   "clarification_ongoing",
    ("resolved",  "ongoing"):   "relapse",
    ("negated",   "ongoing"):   "contradiction",
    ("negated",   "ambiguous"): "contradiction_uncertain",
    ("ongoing",   "ambiguous"): "uncertainty_emerging",
}


@dataclass
class TrajectoryPoint:
    sentence_idx: int
    status: str
    confidence: float
    text: str


@dataclass
class StatusTrajectory:
    condition: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    final_status: str = "ambiguous"
    final_confidence: float = 0.25
    transition_type: str = "stable"
    reason: str = ""


def _confidence_to_llr(confidence: float, status: str) -> dict[str, float]:
    """Convert a single-label confidence into per-label LLRs."""
    p = max(min(confidence, 0.999), 0.001)
    other = (1.0 - p) / (len(LABELS) - 1)
    return {
        lbl: math.log(p / other) if lbl == status else math.log(other / p)
        for lbl in LABELS
    }


def reconcile_trajectory(points: list[TrajectoryPoint]) -> tuple[str, float, str]:
    """
    Compute the trajectory-reconciled final status.

    Uses time-decayed log-evidence accumulation plus a transition bonus
    for clinically significant status changes between consecutive points.

    Returns
    -------
    (status, confidence, transition_type)
    """
    if not points:
        return "ambiguous", 0.25, "stable"

    if len(points) == 1:
        return points[0].status, points[0].confidence, "stable"

    n = len(points)
    log_scores: dict[str, float] = {lbl: 0.0 for lbl in LABELS}

    for i, pt in enumerate(points):
        weight = _DECAY ** (n - 1 - i)
        llrs = _confidence_to_llr(pt.confidence, pt.status)
        for lbl in LABELS:
            log_scores[lbl] += weight * llrs[lbl]

    seen_transition_types: list[str] = []
    for i in range(1, n):
        key = (points[i - 1].status, points[i].status)
        bonus = _TRANSITION_BONUS.get(key)
        if bonus is not None:
            log_scores[points[i].status] += bonus
            seen_transition_types.append(_TRANSITION_TYPE.get(key, "stable"))

    if not seen_transition_types:
        transition_type = "stable"
    elif len(seen_transition_types) == 1:
        transition_type = seen_transition_types[0]
    else:
        transition_type = "multi_transition"

    max_ls = max(log_scores.values())
    exp_scores = {lbl: math.exp(ls - max_ls) for lbl, ls in log_scores.items()}
    total = sum(exp_scores.values())
    posterior = {lbl: exp_scores[lbl] / total for lbl in LABELS}

    final_status = max(posterior, key=posterior.__getitem__)
    final_confidence = posterior[final_status]

    return final_status, final_confidence, transition_type


def build_trajectory(
    entity_text: str,
    sentences: list,
    classify_fn: Callable[[str], dict],
) -> StatusTrajectory:
    """
    Find all sentences mentioning entity_text, classify each, and
    compute the trajectory-reconciled final status.

    Parameters
    ----------
    entity_text : str
        The entity string to search for (word-boundary case-insensitive match).
    sentences : list[Sentence]
        Ordered list of sentences from the containing section.
    classify_fn : Callable
        Function mapping text → {"status": str, "confidence": float, ...}.

    Returns
    -------
    StatusTrajectory
        All trajectory points plus the reconciled final status and transition type.
    """
    traj = StatusTrajectory(condition=entity_text)
    pattern = re.compile(r'\b' + re.escape(entity_text) + r'\b', re.IGNORECASE)

    for idx, sent in enumerate(sentences):
        if not pattern.search(sent.text):
            continue
        clf = classify_fn(sent.text)
        traj.points.append(TrajectoryPoint(
            sentence_idx=idx,
            status=clf["status"],
            confidence=clf["confidence"],
            text=sent.text,
        ))

    if not traj.points:
        traj.reason = "No sentences found mentioning this entity."
        return traj

    final_status, final_conf, transition_type = reconcile_trajectory(traj.points)
    traj.final_status = final_status
    traj.final_confidence = final_conf
    traj.transition_type = transition_type
    traj.reason = _format_reason(traj)
    return traj


def _format_reason(traj: StatusTrajectory) -> str:
    n = len(traj.points)
    if n == 1:
        return f"Single mention: {traj.points[0].status} ({traj.points[0].confidence:.0%})"
    statuses = " → ".join(p.status for p in traj.points)
    if traj.transition_type == "stable":
        return f"Stable across {n} mentions: {statuses}"
    type_label = traj.transition_type.replace("_", " ")
    return f"Trajectory ({type_label}) across {n} mentions: {statuses} → {traj.final_status}"
