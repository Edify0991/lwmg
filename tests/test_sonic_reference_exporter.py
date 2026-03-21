import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lwmg.sonic_io.sonic_reference_exporter import SonicReferenceExporter


def test_sonic_reference_exporter_headers_and_files(tmp_path: Path) -> None:
    data = {
        "joint_pos": torch.zeros(10, 29),
        "joint_vel": torch.zeros(10, 29),
        "body_pos": torch.zeros(10, 3),
        "body_quat": torch.zeros(10, 4),
    }
    clip_dir = SonicReferenceExporter().export(tmp_path, data, clip_name="clip_test")
    assert (clip_dir / "joint_pos.csv").exists()
    assert (clip_dir / "metadata.txt").exists()

    with (clip_dir / "joint_pos.csv").open() as f:
        header = next(csv.reader(f))
    assert header[0] == "joint_0"
    assert len(header) == 29


def test_sonic_reference_exporter_shape_checks(tmp_path: Path) -> None:
    bad = {
        "joint_pos": torch.zeros(10, 28),
        "joint_vel": torch.zeros(10, 28),
        "body_pos": torch.zeros(10, 3),
        "body_quat": torch.zeros(10, 4),
    }
    with pytest.raises(ValueError):
        SonicReferenceExporter().export(tmp_path, bad)
