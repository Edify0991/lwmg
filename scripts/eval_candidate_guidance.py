from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.guidance.candidate_ranker import CandidateRanker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ranker = CandidateRanker(top_k=cfg["eval"].get("num_candidates", 8))
    n = 32
    metrics = {
        "task_progress": torch.rand(n),
        "tracking_feasibility": torch.rand(n),
        "hard_risk": torch.rand(n),
        "soft_risk": torch.rand(n),
        "smoothness": torch.rand(n),
        "torque_penalty": torch.rand(n),
    }
    idx = ranker.rank(metrics)
    print("top candidate indices:", idx.tolist())


if __name__ == "__main__":
    main()
