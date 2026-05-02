"""
Tests for status trajectory tracking (src/trajectory.py).
"""

from src.trajectory import (
    build_trajectory,
    reconcile_trajectory,
    StatusTrajectory,
    TrajectoryPoint,
    LABELS,
)
from src.sentence_splitter import split_sentences
from src.classifier import classify_condition_status


# ---------------------------------------------------------------------------
# reconcile_trajectory
# ---------------------------------------------------------------------------

class TestReconcileTrajectory:

    def test_empty_returns_ambiguous(self):
        status, conf, tt = reconcile_trajectory([])
        assert status == "ambiguous"
        assert tt == "stable"

    def test_single_point_returns_same_status(self):
        pts = [TrajectoryPoint(0, "resolved", 0.90, "")]
        status, conf, tt = reconcile_trajectory(pts)
        assert status == "resolved"
        assert tt == "stable"

    def test_stable_ongoing_two_points(self):
        pts = [
            TrajectoryPoint(0, "ongoing", 0.85, ""),
            TrajectoryPoint(1, "ongoing", 0.90, ""),
        ]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "ongoing"
        assert tt == "stable"

    def test_resolution_transition(self):
        pts = [
            TrajectoryPoint(0, "ongoing", 0.80, ""),
            TrajectoryPoint(1, "resolved", 0.85, ""),
        ]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "resolved"
        assert tt == "resolution"

    def test_relapse_transition(self):
        pts = [
            TrajectoryPoint(0, "resolved", 0.85, ""),
            TrajectoryPoint(1, "ongoing", 0.80, ""),
        ]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "ongoing"
        assert tt == "relapse"

    def test_contradiction_transition(self):
        pts = [
            TrajectoryPoint(0, "negated", 0.90, ""),
            TrajectoryPoint(1, "ongoing", 0.80, ""),
        ]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "ongoing"
        assert tt == "contradiction"

    def test_clarification_resolved(self):
        pts = [
            TrajectoryPoint(0, "ambiguous", 0.70, ""),
            TrajectoryPoint(1, "resolved", 0.85, ""),
        ]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "resolved"
        assert tt == "clarification_resolved"

    def test_clarification_ongoing(self):
        pts = [
            TrajectoryPoint(0, "ambiguous", 0.70, ""),
            TrajectoryPoint(1, "ongoing", 0.85, ""),
        ]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "ongoing"
        assert tt == "clarification_ongoing"

    def test_recency_dominates_three_points(self):
        pts = [
            TrajectoryPoint(0, "ongoing", 0.85, ""),
            TrajectoryPoint(1, "ongoing", 0.85, ""),
            TrajectoryPoint(2, "resolved", 0.90, ""),
        ]
        status, _, _ = reconcile_trajectory(pts)
        assert status == "resolved"

    def test_very_strong_early_holds_against_weak_recent(self):
        # 0.98 ongoing early vs 0.40 resolved late — early dominates;
        # the time-decay system does not blindly favour recency over evidence strength
        pts = [
            TrajectoryPoint(0, "ongoing", 0.98, ""),
            TrajectoryPoint(1, "resolved", 0.40, ""),
        ]
        status, _, _ = reconcile_trajectory(pts)
        assert status == "ongoing"

    def test_multi_transition(self):
        pts = [
            TrajectoryPoint(0, "ongoing", 0.80, ""),
            TrajectoryPoint(1, "resolved", 0.85, ""),
            TrajectoryPoint(2, "ongoing", 0.80, ""),
        ]
        _, _, tt = reconcile_trajectory(pts)
        assert tt == "multi_transition"

    def test_confidence_in_unit_range(self):
        pts = [
            TrajectoryPoint(0, "ongoing", 0.80, ""),
            TrajectoryPoint(1, "resolved", 0.85, ""),
        ]
        _, conf, _ = reconcile_trajectory(pts)
        assert 0.0 <= conf <= 1.0

    def test_stable_same_label_four_points(self):
        pts = [TrajectoryPoint(i, "negated", 0.90, "") for i in range(4)]
        status, _, tt = reconcile_trajectory(pts)
        assert status == "negated"
        assert tt == "stable"


# ---------------------------------------------------------------------------
# build_trajectory
# ---------------------------------------------------------------------------

class TestBuildTrajectory:

    def test_entity_not_found_returns_empty(self):
        sentences = split_sentences("Patient has no fever.")
        traj = build_trajectory("pneumonia", sentences, classify_condition_status)
        assert len(traj.points) == 0

    def test_single_mention(self):
        sentences = split_sentences("Patient has diabetes.")
        traj = build_trajectory("diabetes", sentences, classify_condition_status)
        assert len(traj.points) == 1
        assert traj.transition_type == "stable"

    def test_resolution_trajectory(self):
        sentences = split_sentences(
            "Patient has cough. Cough resolved after antibiotic treatment."
        )
        traj = build_trajectory("cough", sentences, classify_condition_status)
        assert len(traj.points) >= 2
        assert traj.final_status == "resolved"
        assert traj.transition_type == "resolution"

    def test_relapse_trajectory(self):
        sentences = split_sentences(
            "History of diabetes. Diabetes is currently active."
        )
        traj = build_trajectory("diabetes", sentences, classify_condition_status)
        assert len(traj.points) >= 2
        assert traj.final_status == "ongoing"
        assert traj.transition_type == "relapse"

    def test_contradiction_trajectory(self):
        sentences = split_sentences(
            "No evidence of hypertension. Hypertension is now presenting."
        )
        traj = build_trajectory("hypertension", sentences, classify_condition_status)
        assert len(traj.points) >= 2
        assert traj.transition_type in ("contradiction", "multi_transition")

    def test_stable_ongoing(self):
        sentences = split_sentences(
            "Patient has hypertension. Hypertension is currently ongoing."
        )
        traj = build_trajectory("hypertension", sentences, classify_condition_status)
        assert traj.final_status == "ongoing"

    def test_case_insensitive_matching(self):
        sentences = split_sentences(
            "Patient has Hypertension. Hypertension is stable."
        )
        traj = build_trajectory("hypertension", sentences, classify_condition_status)
        assert len(traj.points) >= 2

    def test_condition_field_preserved(self):
        sentences = split_sentences("Patient has chest pain.")
        traj = build_trajectory("chest pain", sentences, classify_condition_status)
        assert traj.condition == "chest pain"

    def test_reason_nonempty_for_single_mention(self):
        sentences = split_sentences("Patient has asthma.")
        traj = build_trajectory("asthma", sentences, classify_condition_status)
        assert traj.reason != ""

    def test_reason_contains_transition_type(self):
        sentences = split_sentences(
            "History of diabetes. Diabetes is currently active."
        )
        traj = build_trajectory("diabetes", sentences, classify_condition_status)
        assert "relapse" in traj.reason.lower() or "trajectory" in traj.reason.lower()

    def test_final_confidence_in_unit_range(self):
        sentences = split_sentences(
            "Patient has cough. Cough resolved after treatment."
        )
        traj = build_trajectory("cough", sentences, classify_condition_status)
        assert 0.0 <= traj.final_confidence <= 1.0

    def test_points_in_sentence_order(self):
        sentences = split_sentences(
            "Cough was resolved. Cough is now ongoing."
        )
        traj = build_trajectory("cough", sentences, classify_condition_status)
        if len(traj.points) >= 2:
            assert traj.points[0].sentence_idx < traj.points[1].sentence_idx


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestTrajectoryPipelineIntegration:

    def test_resolution_across_sentences(self):
        from src.pipeline import process_note
        note = (
            "History of Present Illness:\n"
            "Patient has cough. "
            "Cough resolved after antibiotic treatment.\n"
        )
        result = process_note(note)
        cough = [c for c in result.conditions if "cough" in c.condition.lower()]
        assert cough, "Expected 'cough' to be detected"
        assert cough[0].status == "resolved", (
            f"Expected resolved via trajectory, got {cough[0].status}"
        )

    def test_relapse_sets_ongoing(self):
        from src.pipeline import process_note
        note = (
            "Past Medical History:\n"
            "History of diabetes. Diabetes is currently active.\n"
        )
        result = process_note(note)
        diabetes = [c for c in result.conditions if "diabetes" in c.condition.lower()]
        assert diabetes, "Expected 'diabetes' to be detected"
        assert diabetes[0].status == "ongoing", (
            f"Expected ongoing via relapse trajectory, got {diabetes[0].status}"
        )

    def test_single_mention_unaffected(self):
        from src.pipeline import process_note
        note = "History of Present Illness:\nPatient presents with chest pain.\n"
        result = process_note(note)
        cp = [c for c in result.conditions if "chest pain" in c.condition.lower()]
        assert cp, "Expected 'chest pain'"
        assert cp[0].trajectory is None  # single mention — no trajectory object

    def test_multi_mention_sets_trajectory(self):
        from src.pipeline import process_note
        note = (
            "History of Present Illness:\n"
            "Patient has cough. Cough resolved after treatment.\n"
        )
        result = process_note(note)
        cough = [c for c in result.conditions if "cough" in c.condition.lower()]
        assert cough, "Expected 'cough'"
        assert cough[0].trajectory is not None
        assert len(cough[0].trajectory.points) >= 2
