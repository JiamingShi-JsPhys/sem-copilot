"""Contract tests: contracts/gate_result.py GateResult structure."""
from __future__ import annotations

import json

from contracts import (EvidenceLevel, GateResult, GateStatus, fail_gate,
                       pass_gate, unverified_gate, weakest)


def test_gate_result_fields_and_roundtrip():
    g = fail_gate("reference-auditor", EvidenceLevel.EXACT,
                  reason="DOI 10.9999/fake not found",
                  artifacts=["process/gates/x.json"])
    d = g.to_dict()
    # blueprint section 2 exact field set (+ timestamp).  The optional
    # structured ``data`` payload is omitted from to_dict() when empty, so
    # a default gate carries exactly these six keys.
    assert set(d) == {"skill", "status", "evidence_level", "reason",
                      "artifacts", "timestamp"}
    assert d["skill"] == "reference-auditor"
    assert d["status"] == "FAIL"
    assert d["evidence_level"] == "EXACT"
    assert GateResult.from_dict(d) == g
    # JSON string round trip
    assert GateResult.from_dict(json.loads(json.dumps(d))) == g

    # A gate carrying a structured data payload must round-trip that field
    # verbatim (e.g. sample size / auto-computed mediation effects).
    g2 = GateResult("sem-copilot", GateStatus.PASS, EvidenceLevel.EMPIRICAL,
                    reason="ok", artifacts=[],
                    data={"n_obs": 10, "mediation": [{"x": "X", "y": "Y"}]})
    d2 = g2.to_dict()
    assert "data" in d2 and d2["data"]["n_obs"] == 10
    assert GateResult.from_dict(d2) == g2
    assert GateResult.from_dict(json.loads(json.dumps(d2))) == g2


def test_status_and_evidence_enums():
    assert [s.value for s in GateStatus] == ["PASS", "FAIL", "UNVERIFIED"]
    assert [e.value for e in EvidenceLevel] == [
        "EXACT", "SYMBOLIC", "EMPIRICAL", "UNVERIFIED"]
    # WARN is folded into UNVERIFIED — only FAIL hard-blocks
    assert GateStatus("UNVERIFIED").value == "UNVERIFIED"
    assert not hasattr(GateStatus, "WARN")


def test_weakest_evidence():
    order = [EvidenceLevel.EXACT, EvidenceLevel.SYMBOLIC,
             EvidenceLevel.EMPIRICAL, EvidenceLevel.UNVERIFIED]
    assert weakest(order) == EvidenceLevel.UNVERIFIED
    assert weakest([EvidenceLevel.EXACT, EvidenceLevel.SYMBOLIC]) \
        == EvidenceLevel.SYMBOLIC
    assert weakest([EvidenceLevel.EXACT]) == EvidenceLevel.EXACT


def test_helpers():
    assert pass_gate("formula-validator", EvidenceLevel.SYMBOLIC).status \
        == GateStatus.PASS
    assert unverified_gate("video-tracker", "fft inconclusive").status \
        == GateStatus.UNVERIFIED
    bad = fail_gate("data-profiler", EvidenceLevel.EMPIRICAL)
    assert bad.status == GateStatus.FAIL and bad.artifacts == []
