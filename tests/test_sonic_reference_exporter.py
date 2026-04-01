import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lwmg.sonic_io.reference_source_loader import load_reference_source
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


def _make_lafan_like_csv(path: Path) -> None:
    rows = 3
    data = np.zeros((rows, 36), dtype=np.float32)

    data[:, 0] = [0.0, 0.1, 0.2]
    data[:, 1] = [0.0, 0.0, 0.0]
    data[:, 2] = [0.8, 0.8, 0.8]

    # quat xyzw
    data[:, 3] = 0.0
    data[:, 4] = 0.0
    data[:, 5] = 0.0
    data[:, 6] = 1.0

    for j in range(29):
        data[:, 7 + j] = np.array([j, j + 0.1, j + 0.2], dtype=np.float32)

    np.savetxt(path, data, delimiter=",", fmt="%.6f")


def test_reference_source_loader_lafan1_csv_and_resample(tmp_path: Path) -> None:
    csv_path = tmp_path / "walk.csv"
    _make_lafan_like_csv(csv_path)

    data = load_reference_source(
        {
            "type": "lafan1_retarget_csv",
            "path": str(csv_path),
            "fps": 30,
            "has_header": False,
            "quat_format": "xyzw",
        },
        base_dir=tmp_path,
        expected_joints=29,
        target_hz=60,
    )

    assert tuple(data.joint_pos.shape) == (5, 29)
    assert tuple(data.joint_vel.shape) == (5, 29)
    assert tuple(data.body_pos.shape) == (5, 3)
    assert tuple(data.body_quat.shape) == (5, 4)
    assert torch.allclose(
        data.body_quat[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        atol=1e-5,
    )


def test_reference_source_loader_generated_npz_without_joint_vel(tmp_path: Path) -> None:
    n = 4
    npz_path = tmp_path / "gen_refs.npz"
    np.savez(
        npz_path,
        joint_pos=np.linspace(0.0, 1.0, n * 29, dtype=np.float32).reshape(n, 29),
        body_pos=np.zeros((n, 3), dtype=np.float32),
        body_quat=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
    )

    data = load_reference_source(
        {
            "type": "generated_npz",
            "path": str(npz_path),
            "fps": 50,
            "quat_format": "wxyz",
            "keys": {
                "joint_pos": "joint_pos",
                "body_pos": "body_pos",
                "body_quat": "body_quat",
            },
        },
        base_dir=tmp_path,
        expected_joints=29,
        target_hz=50,
    )

    assert tuple(data.joint_pos.shape) == (n, 29)
    assert tuple(data.joint_vel.shape) == (n, 29)
    assert tuple(data.body_pos.shape) == (n, 3)
    assert tuple(data.body_quat.shape) == (n, 4)


def test_reference_source_loader_generic_csv_with_named_columns(tmp_path: Path) -> None:
    n = 5
    cols = ["px", "py", "pz", "qw", "qx", "qy", "qz"]
    cols += [f"j{i}" for i in range(29)]
    cols += [f"v{i}" for i in range(29)]

    data = np.zeros((n, len(cols)), dtype=np.float32)
    data[:, 0] = np.linspace(0.0, 0.2, n)
    data[:, 3] = 1.0
    for i in range(29):
        data[:, 7 + i] = i
        data[:, 7 + 29 + i] = i * 0.1

    csv_path = tmp_path / "generic.csv"
    pd.DataFrame(data, columns=cols).to_csv(csv_path, index=False)

    loaded = load_reference_source(
        {
            "type": "generic_csv",
            "path": str(csv_path),
            "has_header": True,
            "fps": 50,
            "quat_format": "wxyz",
            "root_pos_cols": ["px", "py", "pz"],
            "root_quat_cols": ["qw", "qx", "qy", "qz"],
            "joint_pos_cols": [f"j{i}" for i in range(29)],
            "joint_vel_cols": [f"v{i}" for i in range(29)],
        },
        base_dir=tmp_path,
        expected_joints=29,
        target_hz=50,
    )

    assert tuple(loaded.joint_pos.shape) == (n, 29)
    assert tuple(loaded.joint_vel.shape) == (n, 29)
    assert tuple(loaded.body_pos.shape) == (n, 3)
    assert tuple(loaded.body_quat.shape) == (n, 4)
