from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


_META_COLUMNS = {"index", "time_ms", "time_realtime_ms", "time_monotonic_ms", "ros_timestamp"}


def _load_replay_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Replay trace not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _load_vector_csv(path: Path, prefixes: tuple[str, ...]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing SONIC CSV: {path}")

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc

        rows = [row for row in reader if row]

    cols = [i for i, c in enumerate(header) if any(c.startswith(p) for p in prefixes)]
    if not cols:
        cols = [i for i, c in enumerate(header) if c not in _META_COLUMNS]
    if not cols:
        raise ValueError(f"No vector columns found in {path}")

    if not rows:
        return np.zeros((0, len(cols)), dtype=np.float32)

    values = np.asarray([[float(r[i]) for i in cols] for r in rows], dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _metric(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.square(diff).mean())),
        "max_abs": float(abs_diff.max()),
    }


def _topk_joint_mae(a: np.ndarray, b: np.ndarray, k: int) -> list[tuple[int, float]]:
    diff = np.abs(a - b)
    mae_joint = diff.mean(axis=0)
    kk = min(k, mae_joint.shape[0])
    idx = np.argsort(mae_joint)[::-1][:kk]
    return [(int(i), float(mae_joint[i])) for i in idx]


def _compare_pair(
    name: str,
    replay: np.ndarray,
    sonic: np.ndarray,
    *,
    max_frames: int | None,
    topk: int,
) -> None:
    if replay.ndim != 2 or sonic.ndim != 2:
        raise ValueError(f"{name}: expected rank-2 arrays, got replay={replay.shape}, sonic={sonic.shape}")

    n = min(replay.shape[0], sonic.shape[0])
    if max_frames is not None:
        n = min(n, int(max_frames))
    if n <= 0:
        raise ValueError(f"{name}: no overlapping frames")

    d = min(replay.shape[1], sonic.shape[1])
    if d <= 0:
        raise ValueError(f"{name}: no overlapping dimensions")

    a = replay[:n, :d]
    b = sonic[:n, :d]
    m = _metric(a, b)
    print(
        f"[{name}] frames={n} dims={d} "
        f"mae={m['mae']:.6f} rmse={m['rmse']:.6f} max_abs={m['max_abs']:.6f}"
    )
    top = _topk_joint_mae(a, b, k=topk)
    top_s = ", ".join([f"j{i}={v:.6f}" for i, v in top])
    print(f"[{name}] top_joint_mae: {top_s}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare replay trace NPZ against SONIC logger CSV outputs.")
    parser.add_argument("--replay-trace", type=Path, required=True, help="Path to replay trace NPZ from replay_motion.py --trace-out")
    parser.add_argument("--sonic-log-dir", type=Path, required=True, help="SONIC logger directory containing q.csv/dq.csv/action.csv")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max number of frames to compare")
    parser.add_argument("--topk", type=int, default=8, help="Top-K joints to print by MAE")
    args = parser.parse_args()

    replay = _load_replay_trace(args.replay_trace)

    required_keys = ["q_meas_abs_pre", "dq_meas_pre", "mapped_action"]
    missing = [k for k in required_keys if k not in replay]
    if missing:
        raise KeyError(
            "Replay trace missing required keys: "
            f"{missing}. Regenerate with replay_motion.py --trace-out."
        )

    sonic_q = _load_vector_csv(args.sonic_log_dir / "q.csv", prefixes=("q_",))
    sonic_dq = _load_vector_csv(args.sonic_log_dir / "dq.csv", prefixes=("dq_",))
    sonic_action = _load_vector_csv(args.sonic_log_dir / "action.csv", prefixes=("act_", "action_"))

    print(f"replay_trace: {args.replay_trace}")
    print(f"sonic_log_dir: {args.sonic_log_dir}")

    _compare_pair(
        "q_abs",
        replay=np.asarray(replay["q_meas_abs_pre"], dtype=np.float32),
        sonic=np.asarray(sonic_q, dtype=np.float32),
        max_frames=args.max_frames,
        topk=args.topk,
    )
    _compare_pair(
        "dq",
        replay=np.asarray(replay["dq_meas_pre"], dtype=np.float32),
        sonic=np.asarray(sonic_dq, dtype=np.float32),
        max_frames=args.max_frames,
        topk=args.topk,
    )
    _compare_pair(
        "action_target",
        replay=np.asarray(replay["mapped_action"], dtype=np.float32),
        sonic=np.asarray(sonic_action, dtype=np.float32),
        max_frames=args.max_frames,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
