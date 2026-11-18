import hashlib
import json
import shutil
from pathlib import Path

import pytest

from donka_scorecard.builder import build


ROOT = Path(__file__).parents[1]


def _config(tmp_path: Path) -> Path:
    shutil.copy(ROOT / "examples" / "sample-applications.csv", tmp_path / "data.csv")
    config = json.loads((ROOT / "examples" / "config.json").read_text(encoding="utf-8"))
    config["data"] = "data.csv"
    config["output"] = "packs"
    config["report"] = "report.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _pack_hash(path: Path) -> dict[str, str]:
    return {file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def test_build_writes_pack_report_and_python_scored_oot_scenarios(tmp_path):
    config = _config(tmp_path)
    pack = build(config)
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    scenarios = json.loads((pack / "scenarios.json").read_text(encoding="utf-8"))
    graph = json.loads((pack / "decisions" / "scorecard.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert manifest["format"] == 1
    assert manifest["decisions"] == ["scorecard"]
    assert len(scenarios) == 8
    assert all(scenario["match"] == "exact" and "score" in scenario["expected"] for scenario in scenarios)
    assert any(scenario["input"]["monthsOnBook"] == -1 for scenario in scenarios)
    assert any(scenario["input"]["monthlyDebt"] == -999 for scenario in scenarios)
    assert any(scenario["input"]["monthlyDebt"] is None for scenario in scenarios)
    assert {node["type"] for node in graph["nodes"]} >= {"inputNode", "decisionTableNode", "expressionNode", "outputNode"}
    assert (pack / "guide.md").is_file() and (pack / "guide.fr.md").is_file()
    assert not any(file.name.endswith(".csv") for file in pack.rglob("*"))
    assert {"gini", "ks"} <= report["out_of_time_metrics"].keys()
    assert report["population_stability"]["score"] is not None
    assert all("independent_iv_total" in feature for feature in report["features"])
    assert "accepted-only" in " ".join(report["known_limits"]).lower()


def test_build_output_is_reproducible(tmp_path):
    config = _config(tmp_path)
    first = build(config)
    first_hash = _pack_hash(first)
    second = build(config)
    assert _pack_hash(second) == first_hash


def test_too_few_bad_outcomes_stops_with_clear_message(tmp_path):
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target"]["minimum_bad_outcomes"] = 10000
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="too few bad outcomes: found .* need at least 10000"):
        build(config_path)
