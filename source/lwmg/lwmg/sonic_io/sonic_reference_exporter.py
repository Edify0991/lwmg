from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from .sonic_motion_format import SonicMotionFormat


class SonicReferenceExporter:
    def __init__(self, fmt: SonicMotionFormat | None = None) -> None:
        self.fmt = fmt or SonicMotionFormat()

    def export(self, out_dir: Path, data: Dict[str, torch.Tensor]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, file_name in {
            "joint_pos": "joint_pos.csv",
            "joint_vel": "joint_vel.csv",
            "body_pos": "body_pos.csv",
            "body_quat": "body_quat.csv",
        }.items():
            pd.DataFrame(data[key].detach().cpu().numpy()).to_csv(out_dir / file_name, index=False)

        metadata = (
            f"frequency_hz={self.fmt.frequency_hz}\n"
            f"joint_order={self.fmt.joint_order}\n"
            f"num_frames={int(data['joint_pos'].shape[0])}\n"
        )
        (out_dir / "metadata.txt").write_text(metadata)
