import mlflow
import pytest

from sentinelml.gate import GateConfig, evaluate_gate, promote, rollback, status
from sentinelml.registry import candidate_tags, register_candidate
from tests.conftest import MODEL, RELAXED, build_registry

GOOD = {"test_fpr": 0.04, "test_pr_auc": 0.99, "test_recall": 0.90, "single_row_latency_ms": 20.0}


def failed(checks):
    return {c.name for c in checks if not c.passed}


def test_all_rules_pass_without_champion():
    checks = evaluate_gate(GOOD, None, GateConfig())
    assert not failed(checks)
    assert any("skipped" in c.detail for c in checks)  # regression rules skipped


def test_fpr_over_budget_is_rejected():
    assert failed(evaluate_gate({**GOOD, "test_fpr": 0.09}, None, GateConfig())) == {"fpr_within_budget"}


def test_slow_model_is_rejected():
    assert failed(evaluate_gate({**GOOD, "single_row_latency_ms": 139.0}, None, GateConfig())) == {"latency_ceiling"}


def test_regression_vs_champion_is_rejected():
    worse = {**GOOD, "test_pr_auc": 0.97, "test_recall": 0.85}
    assert failed(evaluate_gate(worse, GOOD, GateConfig())) == {"no_pr_auc_regression", "no_recall_regression"}


def test_tiny_difference_within_tolerance_passes():
    assert not failed(evaluate_gate({**GOOD, "test_pr_auc": 0.9885}, GOOD, GateConfig()))


def test_missing_metric_fails_closed():
    assert "pr_auc_floor" in failed(evaluate_gate({k: v for k, v in GOOD.items() if k != "test_pr_auc"}, None, GateConfig()))


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return build_registry(tmp_path_factory.mktemp("gate_reg"))


def test_promote_then_second_version_then_rollback(reg):
    uri = reg["uri"]
    first = promote(MODEL, uri, RELAXED)
    assert first.promoted and status(MODEL, uri)["champion"] == first.candidate_version

    # a "retrained" model arrives (same run registered again = identical metrics => no regression)
    v_new = register_candidate(MODEL, reg["lgbm_row"]["run_id"], candidate_tags(reg["lgbm_row"]), uri)
    second = promote(MODEL, uri, RELAXED)
    assert second.promoted and status(MODEL, uri)["champion"] == v_new
    assert status(MODEL, uri)["previous_champion"] == first.candidate_version

    assert rollback(MODEL, uri) == first.candidate_version
    assert status(MODEL, uri)["champion"] == first.candidate_version
    assert status(MODEL, uri)["previous_champion"] is None


def test_rejected_candidate_leaves_champion_untouched(reg):
    uri = reg["uri"]
    before = status(MODEL, uri)["champion"]
    register_candidate(MODEL, reg["lgbm_row"]["run_id"], candidate_tags(reg["lgbm_row"]), uri)
    strict = GateConfig(max_fpr=0.0001, fpr_slack=0.0)
    res = promote(MODEL, uri, strict)
    assert not res.passed and not res.promoted
    assert status(MODEL, uri)["champion"] == before
