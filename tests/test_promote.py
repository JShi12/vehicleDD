import json
import subprocess
import sys

import pytest

from cardd.metrics_schema import validate_metrics_schema
from cardd.promote import (
    apply_promotion,
    build_champion_pointer,
    decide_promotion,
    splice_readme_section,
)


def make_metrics(recall, map50_95=0.5, run_name="run", split="test"):
    return {
        "schema_version": 1,
        "run_name": run_name,
        "config_path": "configs/experiments/01_baseline.yaml",
        "git_sha": "abc1234",
        "mlflow_run_id": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "train_seconds": 100.0,
        "weights_path": "runs/run/weights/best.pt",
        "class_names": ["dent", "scratch"],
        "splits": {
            split: {
                "images": 10,
                "instances": 20,
                "aggregate": {
                    "precision": 0.7, "recall": recall, "map50": 0.6, "map50_95": map50_95,
                },
                "per_class": [
                    {"class_id": 0, "class_name": "dent", "precision": 0.7, "recall": recall,
                     "f1": 0.7, "ap50": 0.6, "ap50_95": map50_95},
                ],
            }
        },
    }


def test_decide_promotion_promotes_on_recall_improvement():
    champion = make_metrics(recall=0.60)
    challenger = make_metrics(recall=0.65)
    decision = decide_promotion(champion, challenger)
    assert decision["promote"] is True
    assert decision["delta_recall"] == pytest.approx(0.05)


def test_decide_promotion_rejects_on_recall_regression():
    champion = make_metrics(recall=0.60)
    challenger = make_metrics(recall=0.55)
    decision = decide_promotion(champion, challenger)
    assert decision["promote"] is False
    assert decision["delta_recall"] == pytest.approx(-0.05)


def test_decide_promotion_respects_tolerance():
    champion = make_metrics(recall=0.60)
    challenger = make_metrics(recall=0.58)  # -0.02, within a 0.03 tolerance
    assert decide_promotion(champion, challenger, recall_tolerance=0.03)["promote"] is True
    assert decide_promotion(champion, challenger, recall_tolerance=0.01)["promote"] is False


def test_decide_promotion_raises_on_missing_split():
    champion = make_metrics(recall=0.60, split="test")
    challenger = make_metrics(recall=0.65, split="val")
    with pytest.raises(ValueError, match="challenger"):
        decide_promotion(champion, challenger, split="test")


def test_build_champion_pointer_shape():
    pointer = build_champion_pointer(
        run_name="03_tuned",
        release_tag="champion-03_tuned-20260101",
        rolling_release_tag="champion",
        weights_url="https://github.com/JShi12/vehicleDD/releases/download/champion/best.pt",
        git_sha="abc1234",
    )
    assert pointer["run_name"] == "03_tuned"
    assert pointer["release_tag"] == "champion-03_tuned-20260101"
    assert pointer["rolling_release_tag"] == "champion"
    assert pointer["metrics_path"] == "models/champion_metrics.json"
    assert "promoted_at" in pointer


def test_splice_readme_section_replaces_between_markers():
    readme = (
        "# Title\n\n## Current champion\n\n"
        "<!-- promote:champion-results:start -->\nold content\n"
        "<!-- promote:champion-results:end -->\n\n"
        "## Next section\n"
    )
    updated = splice_readme_section(readme, "new content")
    assert "old content" not in updated
    assert "new content" in updated
    assert "## Next section" in updated
    assert "<!-- promote:champion-results:start -->" in updated
    assert "<!-- promote:champion-results:end -->" in updated


def test_splice_readme_section_raises_if_markers_missing():
    with pytest.raises(ValueError, match="markers"):
        splice_readme_section("# Title\n\nno markers here\n", "new content")


def test_apply_promotion_writes_all_three_files(tmp_path):
    challenger_metrics_path = tmp_path / "challenger_metrics.json"
    champion_json_path = tmp_path / "models" / "champion.json"
    champion_metrics_path = tmp_path / "models" / "champion_metrics.json"
    readme_path = tmp_path / "README.md"

    challenger = make_metrics(recall=0.7, run_name="03_tuned")
    challenger_metrics_path.write_text(json.dumps(challenger))
    readme_path.write_text(
        "# Title\n\n## Current champion\n\n"
        "<!-- promote:champion-results:start -->\nold\n<!-- promote:champion-results:end -->\n"
    )

    pointer = apply_promotion(
        challenger_metrics_path=challenger_metrics_path,
        champion_json_path=champion_json_path,
        champion_metrics_path=champion_metrics_path,
        readme_path=readme_path,
        release_tag="champion-03_tuned-20260101",
        rolling_release_tag="champion",
        weights_url="https://github.com/JShi12/vehicleDD/releases/download/champion/best.pt",
        git_sha="abc1234",
    )

    assert champion_metrics_path.exists()
    written_metrics = json.loads(champion_metrics_path.read_text())
    validate_metrics_schema(written_metrics)
    assert written_metrics["run_name"] == "03_tuned"

    assert champion_json_path.exists()
    assert json.loads(champion_json_path.read_text()) == pointer

    readme_text = readme_path.read_text()
    assert "old" not in readme_text
    assert "03_tuned" in readme_text


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "cardd.cli.promote", *args],
        capture_output=True, text=True,
    )


def test_cli_check_only_does_not_write_files(tmp_path):
    champion_metrics = tmp_path / "champion_metrics.json"
    challenger_metrics = tmp_path / "challenger_metrics.json"
    champion_metrics.write_text(json.dumps(make_metrics(recall=0.60)))
    challenger_metrics.write_text(json.dumps(make_metrics(recall=0.70, run_name="challenger")))
    decision_out = tmp_path / "decision.json"
    champion_json = tmp_path / "models" / "champion.json"

    result = _run_cli(
        "--check-only",
        "--champion-metrics", str(champion_metrics),
        "--challenger-metrics", str(challenger_metrics),
        "--champion-json", str(champion_json),
        "--decision-out", str(decision_out),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(decision_out.read_text())["promote"] is True
    assert not champion_json.exists()


def test_cli_apply_writes_files(tmp_path):
    champion_metrics = tmp_path / "champion_metrics.json"
    challenger_metrics = tmp_path / "challenger_metrics.json"
    champion_metrics.write_text(json.dumps(make_metrics(recall=0.60)))
    challenger_metrics.write_text(json.dumps(make_metrics(recall=0.70, run_name="challenger")))
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- promote:champion-results:start -->\nold\n"
        "<!-- promote:champion-results:end -->\n"
    )
    champion_json = tmp_path / "models" / "champion.json"

    result = _run_cli(
        "--champion-metrics", str(champion_metrics),
        "--challenger-metrics", str(challenger_metrics),
        "--challenger-weights-url", "https://example.invalid/best.pt",
        "--release-tag", "champion-challenger-20260101",
        "--readme", str(readme),
        "--champion-json", str(champion_json),
    )

    assert result.returncode == 0, result.stderr
    assert champion_json.exists()
    assert json.loads(champion_metrics.read_text())["run_name"] == "challenger"
    assert "challenger" in readme.read_text()
