from __future__ import annotations

import argparse
import copy
import csv
import glob
import re
from pathlib import Path
from typing import Any

import yaml

from lwmg.sonic_io.reference_source_loader import SonicReferenceData, load_reference_source
from lwmg.sonic_io.sonic_motion_format import SonicMotionFormat
from lwmg.sonic_io.sonic_reference_exporter import SonicReferenceExporter


def _resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _sanitize_clip_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return cleaned or "clip_000"


def _default_suffix_for_type(source_type: str) -> str:
    if source_type in {"generated_npz", "npz"}:
        return ".npz"
    if source_type in {"generated_pt", "pt"}:
        return ".pt"
    return ".csv"


def _expand_sources(source_cfg: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    source_type = str(source_cfg.get("type", "lafan1_retarget_csv")).lower().strip()
    if source_type in {"synthetic_zero", "zero"}:
        return [copy.deepcopy(source_cfg)]

    expanded: list[dict[str, Any]] = []

    if "paths" in source_cfg and source_cfg["paths"] is not None:
        for p in source_cfg["paths"]:
            item = copy.deepcopy(source_cfg)
            item["path"] = str(p)
            expanded.append(item)
        return expanded

    if "path_glob" in source_cfg and source_cfg["path_glob"] is not None:
        pattern = str(source_cfg["path_glob"])
        if Path(pattern).is_absolute():
            matches = [Path(p) for p in sorted(glob.glob(pattern))]
        else:
            matches = sorted(base_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched source.path_glob='{pattern}'")
        for p in matches:
            item = copy.deepcopy(source_cfg)
            item["path"] = str(p)
            expanded.append(item)
        return expanded

    if "path" not in source_cfg:
        raise ValueError("source.path (or source.paths/source.path_glob) is required")

    path = _resolve_path(source_cfg["path"], base_dir)
    if path.is_dir():
        suffix = _default_suffix_for_type(source_type)
        pattern = str(source_cfg.get("dir_glob", f"*{suffix}"))
        matches = sorted(path.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files under directory {path} with pattern '{pattern}'")
        for p in matches:
            item = copy.deepcopy(source_cfg)
            item["path"] = str(p)
            expanded.append(item)
        return expanded

    return [copy.deepcopy(source_cfg)]


def _pick_clip_name(
    export_cfg: dict[str, Any],
    *,
    idx: int,
    total: int,
    source_cfg: dict[str, Any],
) -> str:
    source_type = str(source_cfg.get("type", "lafan1_retarget_csv")).lower().strip()
    source_path = source_cfg.get("path", None)
    stem = "synthetic" if source_path is None else Path(str(source_path)).stem

    if total == 1:
        explicit = export_cfg.get("clip_name", None)
        if explicit is not None:
            return _sanitize_clip_name(str(explicit))
        return "clip_000"

    template = str(export_cfg.get("clip_name_template", "clip_{idx:03d}_{stem}"))
    clip_name = template.format(idx=idx, stem=stem, type=source_type)
    return _sanitize_clip_name(clip_name)


def _write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = out_dir / "clips_manifest.csv"
    fieldnames = ["clip_name", "clip_dir", "frames", "source_hz", "source_type", "source_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    export_cfg = cfg.get("export", {})
    source_cfg = cfg.get("source", {"type": "synthetic_zero", "frames": 100, "fps": 50})

    frequency_hz = float(export_cfg.get("frequency_hz", 50))
    expected_joints = int(export_cfg.get("expected_joints", 29))
    joint_order = str(export_cfg.get("joint_order", "isaaclab_g1_29"))

    out_dir = _resolve_path(export_cfg.get("out_dir", "outputs/sonic_refs"), Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter = SonicReferenceExporter(
        fmt=SonicMotionFormat(frequency_hz=int(round(frequency_hz)), joint_order=joint_order),
        expected_joints=expected_joints,
    )

    source_items = _expand_sources(source_cfg, base_dir=cfg_path.parent)

    manifest_rows: list[dict[str, Any]] = []
    for idx, source_item in enumerate(source_items):
        data: SonicReferenceData = load_reference_source(
            source_item,
            base_dir=cfg_path.parent,
            expected_joints=expected_joints,
            target_hz=frequency_hz,
        )
        clip_name = _pick_clip_name(export_cfg, idx=idx, total=len(source_items), source_cfg=source_item)
        clip_dir = exporter.export(out_dir, data.to_export_dict(), clip_name=clip_name)

        source_type = str(source_item.get("type", "unknown"))
        source_path = str(source_item.get("path", ""))
        print(
            f"[export_sonic_references] clip={clip_name} frames={data.joint_pos.shape[0]} "
            f"hz={frequency_hz:.2f} source_type={source_type} source={source_path}"
        )

        manifest_rows.append(
            {
                "clip_name": clip_name,
                "clip_dir": str(clip_dir),
                "frames": int(data.joint_pos.shape[0]),
                "source_hz": float(data.source_hz),
                "source_type": source_type,
                "source_path": source_path,
            }
        )

    _write_manifest(out_dir, manifest_rows)
    print(f"[export_sonic_references] exported {len(manifest_rows)} clip(s) to {out_dir}")


if __name__ == "__main__":
    main()
