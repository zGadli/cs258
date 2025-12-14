"""Train and evaluate DQN agents for the RSA environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rsaenv import make_env

BASE_DIR = Path(__file__).parent
DEFAULT_TRAIN_DIR = BASE_DIR / "data" / "train"
DEFAULT_EVAL_DIR = BASE_DIR / "data" / "eval"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs"


def moving_average(data: List[float], window: int) -> List[float]:
    if not data:
        return []
    window = max(1, window)
    out: List[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(data[start : i + 1])))
    return out


class EpisodeStatsCallback(BaseCallback):
    """Collect per-episode rewards and block rates from env infos."""

    def __init__(self, window: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.ep_rewards: List[float] = []
        self.block_rates: List[float] = []
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._episode_reward = 0.0
        self._episode_blocks = 0
        self._episode_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        # vectorized env support; assume single env for this project
        self._episode_reward += float(rewards[0])
        self._episode_blocks += sum(info.get("blocked", 0) for info in infos)
        self._episode_steps += 1

        if dones[0]:
            block_rate = self._episode_blocks / max(self._episode_steps, 1)
            self.ep_rewards.append(self._episode_reward)
            self.block_rates.append(block_rate)
            self._reset_counters()
        return True


def plot_series(values: List[float], ylabel: str, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_agent(
    capacity: int,
    train_dir: Path,
    output_dir: Path,
    episodes: int,
    seed: int,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    save_model: bool = True,
) -> Tuple[Path | None, EpisodeStatsCallback]:
    env = make_env(dataset_dir=train_dir, capacity=capacity, seed=seed)
    env = Monitor(env)
    stats_cb = EpisodeStatsCallback(window=10)

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=50_000,
        learning_starts=500,
        batch_size=128,
        tau=0.9,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=seed,
    )

    episode_len = env.env.episode_length  # type: ignore[attr-defined]
    total_timesteps = episodes * episode_len
    model.learn(total_timesteps=total_timesteps, callback=stats_cb, progress_bar=True)

    model_path = None
    if save_model:
        model_path = output_dir / f"dqn_cap{capacity}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    reward_ma = moving_average(stats_cb.ep_rewards, stats_cb.window)
    block_ma = moving_average(stats_cb.block_rates, stats_cb.window)
    plot_series(
        reward_ma,
        ylabel="Avg return (last 10 eps)",
        title=f"Learning curve (cap={capacity})",
        path=output_dir / f"learning_curve_cap{capacity}.png",
    )
    plot_series(
        block_ma,
        ylabel="Blocking B (last 10 eps)",
        title=f"Blocking objective (cap={capacity})",
        path=output_dir / f"blocking_curve_cap{capacity}.png",
    )
    env.close()
    return model_path, stats_cb


def evaluate_agent(
    model_path: Path,
    capacity: int,
    eval_dir: Path,
    episodes: int,
    seed: int,
) -> List[float]:
    model = DQN.load(model_path)
    eval_env = make_env(
        dataset_dir=eval_dir,
        capacity=capacity,
        seed=seed,
        file_selection="sequential",
    )

    block_rates: List[float] = []
    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        blocked = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            blocked += info.get("blocked", 0)
            steps += 1
            done = terminated or truncated
        block_rates.append(blocked / max(steps, 1))
    eval_env.close()
    return block_rates


def quick_tune(
    capacity: int,
    train_dir: Path,
    seed: int,
    episodes: int = 50,
    grid: Iterable[Tuple[float, float]] = ((1e-3, 0.99), (5e-4, 0.995), (2e-3, 0.99)),
) -> Tuple[float, float]:
    """Very small hyperparameter sweep over (learning_rate, gamma)."""
    best_cfg = None
    best_score = float("inf")
    for lr, gamma in grid:
        _, stats = train_agent(
            capacity=capacity,
            train_dir=train_dir,
            output_dir=Path("_tune_tmp"),
            episodes=episodes,
            seed=seed,
            learning_rate=lr,
            gamma=gamma,
            save_model=False,
        )
        mean_block = float(np.mean(stats.block_rates[-10:]))
        if mean_block < best_score:
            best_score = mean_block
            best_cfg = (lr, gamma)
    return best_cfg if best_cfg else (1e-3, 0.99)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for RSA.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[20, 10],
        help="Run training for these capacities.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a quick hyperparameter sweep before main training.",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cap in args.capacities:
        cap_dir = args.output_dir / f"cap_{cap}"
        cap_dir.mkdir(parents=True, exist_ok=True)

        lr, gamma = (1e-3, 0.99)
        if args.tune:
            lr, gamma = quick_tune(capacity=cap, train_dir=args.train_dir, seed=args.seed)
            print(f"[cap={cap}] tuned lr={lr}, gamma={gamma}")

        model_path, stats = train_agent(
            capacity=cap,
            train_dir=args.train_dir,
            output_dir=cap_dir,
            episodes=args.episodes,
            seed=args.seed,
            learning_rate=lr,
            gamma=gamma,
        )

        if model_path:
            eval_blocks = evaluate_agent(
                model_path=model_path,
                capacity=cap,
                eval_dir=args.eval_dir,
                episodes=args.eval_episodes,
                seed=args.seed + 1,
            )
            eval_ma = moving_average(eval_blocks, 10)
            plot_series(
                eval_ma,
                ylabel="Blocking B (eval, last 10 eps)",
                title=f"Eval blocking (cap={cap})",
                path=cap_dir / f"eval_block_cap{cap}.png",
            )


if __name__ == "__main__":
    main()


