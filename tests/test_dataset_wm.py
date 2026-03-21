import torch

from lwmg.data.dataset_wm import WMDataset


def test_dataset_wm_len_and_item() -> None:
    ds = WMDataset(torch.randn(5, 3), torch.randn(5, 2))
    assert len(ds) == 5
    item = ds[0]
    assert item.x.shape[0] == 3
