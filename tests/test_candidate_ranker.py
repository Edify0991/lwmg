import torch

from lwmg.guidance.candidate_ranker import CandidateRanker


def test_candidate_ranker_topk() -> None:
    ranker = CandidateRanker(top_k=3)
    n = 6
    idx = ranker.rank(
        {
            "task_progress": torch.linspace(0, 1, n),
            "tracking_feasibility": torch.ones(n),
            "hard_risk": torch.zeros(n),
            "soft_risk": torch.zeros(n),
            "smoothness": torch.zeros(n),
            "torque_penalty": torch.zeros(n),
        }
    )
    assert idx.shape[0] == 3
