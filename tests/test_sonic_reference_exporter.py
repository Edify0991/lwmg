from pathlib import Path

import torch

from lwmg.sonic_io.sonic_reference_exporter import SonicReferenceExporter


def test_sonic_reference_exporter(tmp_path: Path) -> None:
    data = {
        "joint_pos": torch.zeros(10, 12),
        "joint_vel": torch.zeros(10, 12),
        "body_pos": torch.zeros(10, 3),
        "body_quat": torch.zeros(10, 4),
    }
    SonicReferenceExporter().export(tmp_path, data)
    assert (tmp_path / "joint_pos.csv").exists()
    assert (tmp_path / "metadata.txt").exists()
