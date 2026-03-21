import torch

from lwmg.data.dataset_wm import WMDataset


def test_paired_counterfactual_dataset_mode() -> None:
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)
    splits = ["nominal", "loaded", "pair", "pair"]
    pair_ids = [None, None, "a", "a"]
    ds = WMDataset(x, y, splits=splits, pair_ids=pair_ids, mode="pair")
    assert len(ds) == 2
    assert ds[0].pair_id == "a"
