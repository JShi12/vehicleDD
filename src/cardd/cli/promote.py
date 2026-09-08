import argparse
import json
import os
from pathlib import Path

from cardd.promote import apply_promotion, decide_promotion
from cardd.util import git_sha


def main():
    p = argparse.ArgumentParser(
        description="Compare a challenger's metrics.json against the current champion's and, "
                     "unless --check-only, publish the promotion if it wins."
    )
    p.add_argument("--champion-metrics", type=Path, default=Path("models/champion_metrics.json"))
    p.add_argument("--challenger-metrics", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--recall-tolerance", type=float, default=0.0)
    p.add_argument("--check-only", action="store_true", help="Decide only; never write any files.")
    p.add_argument("--challenger-weights-url", default=None, help="Required unless --check-only.")
    p.add_argument("--release-tag", default=None, help="Required unless --check-only.")
    p.add_argument("--rolling-release-tag", default="champion")
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--champion-json", type=Path, default=Path("models/champion.json"))
    p.add_argument("--decision-out", type=Path, default=None)
    p.add_argument("--github-output", type=Path, default=None,
                    help="Defaults to $GITHUB_OUTPUT if set and this flag is omitted.")
    args = p.parse_args()

    with open(args.champion_metrics) as f:
        champion = json.load(f)
    with open(args.challenger_metrics) as f:
        challenger = json.load(f)

    decision = decide_promotion(
        champion, challenger, split=args.split, recall_tolerance=args.recall_tolerance
    )
    print(decision["reason"])

    if args.decision_out:
        args.decision_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.decision_out, "w") as f:
            json.dump(decision, f, indent=2)

    github_output_path = args.github_output
    if github_output_path is None and os.environ.get("GITHUB_OUTPUT"):
        github_output_path = Path(os.environ["GITHUB_OUTPUT"])
    if github_output_path:
        with open(github_output_path, "a") as f:
            f.write(f"promote={'true' if decision['promote'] else 'false'}\n")

    if args.check_only:
        return

    if not decision["promote"]:
        print("Not promoting - skipping apply step.")
        return

    if not args.challenger_weights_url or not args.release_tag:
        p.error(
            "--challenger-weights-url and --release-tag are required to apply a promotion "
            "(pass --check-only if you only want the decision)."
        )

    pointer = apply_promotion(
        challenger_metrics_path=args.challenger_metrics,
        champion_json_path=args.champion_json,
        champion_metrics_path=args.champion_metrics,
        readme_path=args.readme,
        release_tag=args.release_tag,
        rolling_release_tag=args.rolling_release_tag,
        weights_url=args.challenger_weights_url,
        git_sha=git_sha(),
    )
    print(f"Applied promotion: {pointer['run_name']} -> {pointer['release_tag']}")
    print(f"  {args.champion_metrics}, {args.champion_json}, {args.readme} updated.")


if __name__ == "__main__":
    main()
