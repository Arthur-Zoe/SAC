import hashlib
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from sac import Critic, GaussianActor, ReplayBuffer, SACAgent
from sac.evaluate import evaluate
from sac.train import load_config
from sac.utils import set_seed


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def make_agent(auto_alpha=True, hidden_dim=32, device="cpu", double_q=True):
    return SACAgent(3, 1, np.array([-2.0]), np.array([2.0]), device=device,
                    auto_alpha=auto_alpha, hidden_dim=hidden_dim, double_q=double_q)


def fill_buffer(capacity=64):
    buffer = ReplayBuffer(capacity)
    env = gym.make("Pendulum-v1")
    try:
        obs, _ = env.reset(seed=42)
        env.action_space.seed(42)
        for _ in range(capacity):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, float(terminated))
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return buffer


def test_pendulum_actor_critic_shapes():
    env = gym.make("Pendulum-v1")
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (3,)
        actor = GaussianActor(3, 1, env.action_space.low, env.action_space.high)
        batch = torch.as_tensor(np.stack([obs] * 16))
        assert isinstance(actor(batch), torch.distributions.Normal)
        action, log_prob = actor.sample(batch)
        assert action.shape == (16, 1)
        assert log_prob.shape == (16, 1)
        assert torch.isfinite(log_prob).all()
        assert (action >= -2).all() and (action <= 2).all()
        assert Critic(3, 1)(batch, action).shape == (16, 1)
        agent = make_agent()
        assert agent.act(obs).shape == env.action_space.shape == (1,)
        np.testing.assert_array_equal(agent.act(obs, True), agent.act(obs, True))
        assert np.isfinite(env.step(agent.act(obs))[1])
    finally:
        env.close()


def test_multidimensional_action_scaling_and_log_prob():
    actor = GaussianActor(3, 2, [-3, 1], [1, 5], hidden_dim=32)
    obs = torch.randn(16, 3)
    torch.manual_seed(7)
    action, log_prob = actor.sample(obs)
    torch.manual_seed(7)
    dist = actor(obs)
    u = dist.rsample()
    squashed = torch.tanh(u)
    expected = (dist.log_prob(u) - torch.log(2 * (1 - squashed.square()) + 1e-6)).sum(-1, keepdim=True)
    torch.testing.assert_close(log_prob, expected)
    assert action.shape == (16, 2)
    assert log_prob.shape == (16, 1)
    assert (action >= torch.tensor([-3, 1])).all()
    assert (action <= torch.tensor([1, 5])).all()


def test_replay_shapes_and_ring_overwrite():
    buffer = fill_buffer()
    batch = buffer.sample(16)
    assert [tuple(x.shape) for x in batch] == [(16, 3), (16, 1), (16, 1), (16, 3), (16, 1)]
    assert all(x.dtype == torch.float32 and x.device.type == "cpu" for x in batch)
    ring = ReplayBuffer(2)
    obs = np.zeros(3)
    for reward in range(5):
        ring.add(obs, [0], reward, obs, False)
    obs[:] = 99
    assert len(ring) == 2
    assert {transition[2] for transition in ring.storage} == {3, 4}
    assert all(np.all(transition[0] == 0) for transition in ring.storage)


@pytest.mark.parametrize("double_q", [True, False])
@pytest.mark.parametrize("auto_alpha", [True, False])
def test_update_backward_finite_and_soft_target(auto_alpha, double_q):
    set_seed(42)
    agent = make_agent(auto_alpha=auto_alpha, double_q=double_q)
    before_actor = [p.detach().clone() for p in agent.actor.parameters()]
    before_q = [p.detach().clone() for p in agent.q1.parameters()]
    before_target = [p.detach().clone() for p in agent.target_q1.parameters()]
    metrics = agent.update(fill_buffer(), 16)
    assert all(torch.isfinite(value).all() and not value.requires_grad for value in metrics.values())
    assert any(not torch.equal(before, after) for before, after in zip(before_actor, agent.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_q, agent.q1.parameters()))
    for network in (agent.actor, agent.q1) + ((agent.q2,) if double_q else ()):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    assert any(p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None and not p.requires_grad for p in agent.target_q1.parameters())
    for before, q, target in zip(before_target, agent.q1.parameters(), agent.target_q1.parameters()):
        torch.testing.assert_close(target, (1 - agent.tau) * before + agent.tau * q)
    if auto_alpha:
        assert agent.log_alpha.grad is not None and torch.isfinite(agent.log_alpha.grad)
    else:
        assert agent.get_alpha().item() == pytest.approx(0.2)
        assert metrics["loss/alpha"].item() == 0


@pytest.mark.parametrize("double_q", [True, False])
def test_terminal_and_time_limit_bootstrap(monkeypatch, double_q):
    agent = make_agent(double_q=double_q)
    obs = torch.zeros(2, 3)
    action = torch.zeros(2, 1)
    reward = torch.full((2, 1), 3.0)
    done = torch.tensor([[1.0], [0.0]])  # true terminal, then time-limit truncation
    monkeypatch.setattr(agent.actor, "sample", lambda obs: (action, torch.full((2, 1), -0.5)))
    monkeypatch.setattr(agent.target_q1, "forward", lambda obs, action: torch.full((2, 1), 2.0))
    if double_q:
        monkeypatch.setattr(agent.target_q2, "forward", lambda obs, action: torch.full((2, 1), 1.0))
    expected_q = 1.0 if double_q else 2.0
    expected = torch.tensor([[3.0], [3.0 + agent.gamma * (expected_q + 0.2 * 0.5)]])
    recorded_targets = []
    original_loss = torch.nn.functional.mse_loss

    def capture_target(value, target):
        recorded_targets.append(target.clone())
        return original_loss(value, target)

    monkeypatch.setattr("sac.agent.F.mse_loss", capture_target)
    agent.update_critic(obs, action, reward, obs, done)
    assert len(recorded_targets) == (2 if double_q else 1)
    for target in recorded_targets:
        torch.testing.assert_close(target, expected)


def test_evaluation_fixed_seeds_and_default_20(monkeypatch):
    seeds = []

    class OneStepEnv:
        def reset(self, seed):
            seeds.append(seed)
            return np.zeros(3), {}

        def step(self, action):
            return np.zeros(3), float(seeds[-1]), False, True, {}

        def close(self):
            pass

    class DeterministicAgent:
        def act(self, obs, deterministic=False):
            assert deterministic
            return np.zeros(1)

    monkeypatch.setattr("sac.evaluate.gym.make", lambda name: OneStepEnv())
    mean, std = evaluate(DeterministicAgent(), "ignored")
    assert seeds == list(range(1000, 1020))
    assert mean == np.mean(seeds)
    assert std == np.std(seeds)


def test_pendulum_evaluation_repeatability():
    agent = make_agent()
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    first = evaluate(agent, "Pendulum-v1", episodes=2)
    second = evaluate(agent, "Pendulum-v1", episodes=2)
    assert first == second
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], np_state[1])


@pytest.mark.parametrize("config_name", ["pendulum", "halfcheetah"])
@pytest.mark.parametrize("double_q", [True, False])
def test_configs_and_overrides(config_name, double_q):
    path = ROOT / "configs" / f"{config_name}.yaml"
    assert load_config(path).double_q is True
    config = load_config(path, ["seed=7", "auto_alpha=false", f"double_q={str(double_q).lower()}"])
    assert config.double_q is double_q
    with pytest.raises(ValueError, match="double_q must have type bool"):
        load_config(path, ["double_q=1"])
    assert config.seed == 7 and config.auto_alpha is False
    assert config.eval_episodes == 20
    with pytest.raises(ValueError, match="Unknown config"):
        load_config(ROOT / "configs" / f"{config_name}.yaml", ["typo=1"])


@pytest.mark.parametrize("double_q", [True, False])
def test_training_loop_time_limit_and_local_logs(tmp_path, monkeypatch, double_q):
    from sac.train import train

    real_make = gym.make
    buffers = []

    class RecordingBuffer(ReplayBuffer):
        def __init__(self, capacity):
            super().__init__(capacity)
            buffers.append(self)

    monkeypatch.setattr("sac.train.ReplayBuffer", RecordingBuffer)
    monkeypatch.setattr("sac.train.gym.make", lambda name: real_make(name, max_episode_steps=1))
    config = load_config(ROOT / "configs/pendulum.yaml", [
        f"double_q={str(double_q).lower()}", "total_steps=4", "batch_size=2", "buffer_size=8", "update_after=1",
        "start_steps=1", "updates_per_step=2", "eval_interval=3", "eval_episodes=2",
        "hidden_dim=16", "log_interval=2", "wandb_mode=disabled", f"output_dir={tmp_path}",
    ])
    summary = train(config)
    assert summary["episodes"] == 4
    assert summary["gradient_updates"] == 6
    assert all(transition[-1] == 0.0 for transition in buffers[0].storage)
    rows = [json.loads(line) for line in (Path(summary["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
    assert [row["env_step"] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]["train/episode_length"] == 1
    assert all(key in rows[-1] for key in ("eval/return_mean", "eval/return_std", "loss/critic"))
    assert set(rows[-1]) == {
        "train/episode_return", "train/episode_length", "train/replay_buffer_size", "env_step",
        "eval/return_mean", "eval/return_std", "loss/critic", "critic/q1", "critic/q2",
        "critic/target_q", "loss/actor", "loss/alpha", "policy/alpha", "policy/entropy",
        "policy/log_prob",
    }
    if not double_q:
        assert rows[-1]["critic/q2"] == rows[-1]["critic/q1"]
    assert (Path(summary["run_dir"]) / "actor.pt").is_file()


@pytest.mark.parametrize("kwargs", [{}, {"double_q": True}])
def test_original_backup_and_three_update_equivalence(kwargs):
    """Guard against changing the working script's SAC calculation during extraction."""
    backup = sorted((ROOT / "backups").glob("original_*"))[0]
    for name, digest in json.loads((backup / "SHA256.json").read_text()).items():
        assert hashlib.sha256((backup / name).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location("original_sac", backup / "test_actor_critic.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    set_seed(123)
    old = original.SACAgent(3, 1, [-2.0], [2.0], torch.device("cpu"))
    set_seed(123)
    new = SACAgent(3, 1, [-2.0], [2.0], **kwargs)
    buffer = fill_buffer()
    old_buffer = original.ReplayBuffer(64)
    for transition in buffer.storage:
        old_buffer.add(*transition)
    names = {
        "critic_loss": "loss/critic", "actor_loss": "loss/actor", "alpha_loss": "loss/alpha",
        "q1": "critic/q1", "q2": "critic/q2", "target_q": "critic/target_q",
        "alpha": "policy/alpha", "log_prob": "policy/log_prob",
    }
    for seed in (51, 52, 53):
        set_seed(seed)
        old_metrics = old.update(old_buffer, 16)
        old_rng = torch.get_rng_state().clone()
        set_seed(seed)
        new_metrics = new.update(buffer, 16)
        assert torch.equal(old_rng, torch.get_rng_state())
        for old_key, new_key in names.items():
            assert old_metrics[old_key] == new_metrics[new_key].item()
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            for key, value in getattr(old, name).state_dict().items():
                torch.testing.assert_close(value, getattr(new, name).state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(old.log_alpha, new.log_alpha, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_update():
    assert all(value.device.type == "cuda" for value in make_agent(device="cuda").update(fill_buffer(), 16).values())


@pytest.mark.parametrize("double_q", [True, False])
def test_critic_creation_and_optimizer_ownership(monkeypatch, double_q):
    created = []

    def create_critic(*args, **kwargs):
        network = Critic(*args, **kwargs)
        created.append(network)
        return network

    monkeypatch.setattr("sac.agent.Critic", create_critic)
    agent = make_agent(double_q=double_q)
    assert len(created) == (2 if double_q else 1)
    assert (agent.q2 is None) is (not double_q)
    assert (agent.target_q2 is None) is (not double_q)
    critics = (agent.q1, agent.q2) if double_q else (agent.q1,)
    expected = {id(p) for network in critics for p in network.parameters()}
    actual = {id(p) for group in agent.critic_optimizer.param_groups for p in group["params"]}
    assert actual == expected
    targets = (agent.target_q1, agent.target_q2) if double_q else (agent.target_q1,)
    assert all(not p.requires_grad for network in targets for p in network.parameters())


def test_single_q_actor_and_no_min(monkeypatch):
    agent = make_agent(double_q=False)
    obs = torch.randn(8, 3)
    set_seed(12)
    action, log_prob = agent.actor.sample(obs)
    expected = (agent.get_alpha().detach() * log_prob - agent.q1(obs, action)).mean().detach()

    def forbidden_min(*args, **kwargs):
        pytest.fail("Single-Q must not call torch.min")

    monkeypatch.setattr(torch, "min", forbidden_min)
    set_seed(12)
    loss, actual_log_prob = agent.update_actor(obs)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_log_prob, log_prob, rtol=0, atol=0)
    agent.update(fill_buffer(), 16)
