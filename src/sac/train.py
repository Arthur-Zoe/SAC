import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch
import yaml

from .agent import SACAgent
from .evaluate import evaluate
from .replay_buffer import ReplayBuffer
from .utils import get_device, set_seed


@dataclass
class TrainConfig:
    env_name: str
    seed: int
    total_steps: int
    buffer_size: int
    batch_size: int
    gamma: float
    tau: float
    learning_rate: float
    start_steps: int
    update_after: int
    updates_per_step: int
    eval_interval: int
    eval_episodes: int
    init_alpha: float
    auto_alpha: bool
    hidden_dim: int
    eval_seed_start: int = 1000
    log_interval: int = 100
    device: str = "auto"
    torch_threads: int = 1
    wandb_project: str = "sac-reproduction"
    wandb_mode: str = "offline"
    wandb_group: str = "baseline"
    output_dir: str = "runs"
    double_q: bool = True

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite number")
            elif type(value) is not field.type:
                raise ValueError(f"{field.name} must have type {field.type.__name__}")
        for name in (
            "total_steps", "buffer_size", "batch_size", "updates_per_step", "eval_interval",
            "eval_episodes", "hidden_dim", "log_interval", "torch_threads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.seed, self.eval_seed_start, self.start_steps, self.update_after) < 0:
            raise ValueError("seeds, start_steps and update_after must be nonnegative")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32 for numpy")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must be at least batch_size")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma must be in [0, 1], tau in (0, 1]")
        if self.learning_rate <= 0 or self.init_alpha <= 0:
            raise ValueError("learning_rate and init_alpha must be positive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline or disabled")


def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Config must be a YAML mapping")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {override!r}")
        values[key] = yaml.safe_load(value)
    unknown = values.keys() - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = TrainConfig(**values)
    config.validate()
    return config


def train(config):
    config.validate()
    set_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = get_device(config.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(config.output_dir).resolve() / f"{config.env_name}_seed{config.seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    runtime = {
        "python": platform.python_version(), "torch": torch.__version__,
        "gymnasium": gym.__version__, "numpy": np.__version__,
        "device": str(device), "torch_threads": torch.get_num_threads(),
    }
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"device={device} run_dir={run_dir}", flush=True)
    env, run = None, None
    started = time.perf_counter()
    try:
        env = gym.make(config.env_name)
        if not (
            isinstance(env.observation_space, gym.spaces.Box)
            and isinstance(env.action_space, gym.spaces.Box)
            and len(env.observation_space.shape) == len(env.action_space.shape) == 1
        ):
            raise ValueError("SAC expects vector Box observations and continuous vector Box actions")
        obs, _ = env.reset(seed=config.seed)
        env.action_space.seed(config.seed)
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
            action_low=env.action_space.low, action_high=env.action_space.high,
            device=device, gamma=config.gamma, tau=config.tau,
            learning_rate=config.learning_rate, init_alpha=config.init_alpha,
            auto_alpha=config.auto_alpha, hidden_dim=config.hidden_dim,
            double_q=config.double_q,
        )
        replay_buffer = ReplayBuffer(config.buffer_size)
        if config.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=config.wandb_project, mode=config.wandb_mode,
                config=asdict(config), group=config.wandb_group,
                name=f"{config.env_name}-seed{config.seed}-{stamp}", dir=str(run_dir),
            )
            run.define_metric("env_step")
            for prefix in ("train", "eval", "loss", "critic", "policy"):
                run.define_metric(f"{prefix}/*", step_metric="env_step")

        episode_return, episode_length, episodes, updates = 0.0, 0, 0, 0
        last_info, last_eval = {}, None
        loss_ranges = {}
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log_file:
            for step in range(1, config.total_steps + 1):
                action = env.action_space.sample() if step <= config.start_steps else agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                if not np.isfinite(reward) or not np.isfinite(next_obs).all() or not np.isfinite(action).all():
                    raise FloatingPointError(f"Nonfinite environment transition at env_step={step}")
                # Reset at either ending, bootstrap through TimeLimit truncations only.
                replay_buffer.add(obs, action, reward, next_obs, float(terminated))
                obs = next_obs
                episode_return += reward
                episode_length += 1
                row = {}
                if terminated or truncated:
                    episodes += 1
                    row.update({
                        "train/episode_return": episode_return,
                        "train/episode_length": episode_length,
                    })
                    print(f"step={step} episode={episodes} return={episode_return:.2f} len={episode_length}", flush=True)
                    obs, _ = env.reset()
                    episode_return, episode_length = 0.0, 0

                if step >= config.update_after and len(replay_buffer) >= config.batch_size:
                    for _ in range(config.updates_per_step):
                        last_info = agent.update(replay_buffer, config.batch_size)
                        updates += 1

                if step % config.log_interval == 0 or step == config.total_steps:
                    # Latest gradient update snapshot, not one W&B call per gradient update.
                    row.update({key: value.item() for key, value in last_info.items()})
                    for key, value in row.items():
                        if key.startswith("loss/"):
                            bounds = loss_ranges.setdefault(key, [value, value])
                            bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
                if step % config.eval_interval == 0 or step == config.total_steps:
                    mean_return, std_return = evaluate(
                        agent, config.env_name, config.eval_episodes, config.eval_seed_start
                    )
                    row.update({"eval/return_mean": mean_return, "eval/return_std": std_return})
                    last_eval = {"env_step": step, "mean_return": mean_return, "std_return": std_return}
                    print(f"eval step={step} mean={mean_return:.2f} std={std_return:.2f}", flush=True)
                if row:
                    row.update({"env_step": step, "train/replay_buffer_size": len(replay_buffer)})
                    if not all(np.isfinite(value) for value in row.values()):
                        raise FloatingPointError(f"Nonfinite logged metric at env_step={step}: {row}")
                    log_file.write(json.dumps(row, allow_nan=False) + "\n")
                    log_file.flush()
                    if run is not None:
                        run.log(row, step=step)

        parameters = [p for network in (agent.actor, agent.q1, agent.q2, agent.target_q1, agent.target_q2)
                      if network is not None for p in network.parameters()] + [agent.get_alpha()]
        if not all(torch.isfinite(p).all().item() for p in parameters):
            raise FloatingPointError("Nonfinite final parameters")
        torch.save(agent.actor.state_dict(), run_dir / "actor.pt")
        summary = {
            "env_name": config.env_name, "seed": config.seed, "env_steps": config.total_steps,
            "gradient_updates": updates, "episodes": episodes, "replay_buffer_size": len(replay_buffer),
            "final_eval": last_eval, "final_alpha": agent.get_alpha().item(),
            "logged_loss_ranges": loss_ranges, "final_parameters_finite": True,
            "elapsed_seconds": time.perf_counter() - started, "runtime": runtime,
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if run is not None:
            run.summary.update({"gradient_updates": updates, "final_parameters_finite": True})
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description="Train the original SAC with a YAML experiment config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-name")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = list(args.overrides)
    for key in ("seed", "env_name", "total_steps", "device", "wandb_mode"):
        value = getattr(args, key)
        if value is not None:
            overrides.append(f"{key}={value}")
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
