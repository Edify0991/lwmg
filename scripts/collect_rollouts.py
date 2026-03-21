from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.data.io_utils import save_pickle
from lwmg.data.rollout_buffer import RolloutEpisode, RolloutStep
from lwmg.envs.humanoid_load_env import HumanoidLoadEnv
from lwmg.references.reference_types import ReferenceTarget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    env = HumanoidLoadEnv()
    episode = RolloutEpisode()
    q = torch.zeros(12)
    dq = torch.zeros(12)
    action = torch.zeros(12)
    for _ in range(cfg["collect"].get("num_episodes", 1)):
        out = env.step(q, dq, action)
        episode.steps.append(
            RolloutStep(
                observation=out.observation,
                privileged=out.privileged,
                reference=ReferenceTarget(joint_pos=q.unsqueeze(0), joint_vel=dq.unsqueeze(0)),
                failure_flags=out.flags,
            )
        )
    save_pickle(Path("outputs/demo_rollout.pkl"), episode)
    print("saved outputs/demo_rollout.pkl")


if __name__ == "__main__":
    main()
