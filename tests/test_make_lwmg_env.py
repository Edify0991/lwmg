import pytest

torch = pytest.importorskip("torch")

from lwmg.envs.humanoid_load_env import make_lwmg_env


def test_make_lwmg_env_vectorized_step_and_reset_idx() -> None:
    env = make_lwmg_env({"num_envs": 3, "num_joints": 12}, device="cpu")
    obs = env.reset()
    assert obs["q"].shape == (3, 12)

    action = torch.zeros(3, 12)
    obs, reward, done, _ = env.step(action)
    assert reward.shape[0] == 3

    env_ids = torch.tensor([0, 2], dtype=torch.long)
    env.reset_idx(env_ids)
    assert torch.allclose(env.q[env_ids], torch.zeros_like(env.q[env_ids]))
