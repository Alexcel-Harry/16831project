# =============================================================================
# jump_train.py — Train Humanoid Jump Policy v2d (Windows/Linux/Mac)
# =============================================================================
#
# Usage:
#   python jump_train.py
#   python jump_train.py --timesteps 10000000 --envs 4
#
# Requirements:
#   pip install gymnasium[mujoco] stable-baselines3 torch
#
# Output:
#   humanoid_jumper_v2.zip          — trained PPO model
#   humanoid_jumper_v2_vecnorm.pkl  — observation/reward normalization stats
#
# Both files are needed for evaluation in jump_check.py.
# =============================================================================

import argparse
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from jump_env import JumpRewardWrapper


# =============================================================================
# Callbacks
# =============================================================================
class JumpMetricsCallback(BaseCallback):
    def __init__(self, print_every=20_000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self.jump_heights = []
        self.flight_steps_list = []
        self.ep_count = 0
        self.flight_ep_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.ep_count += 1
            if "jump_height" in info:
                self.jump_heights.append(info["jump_height"])
                self.flight_steps_list.append(info["flight_steps"])
                self.flight_ep_count += 1

        if self.num_timesteps % self.print_every == 0:
            if self.jump_heights:
                recent_h = self.jump_heights[-50:]
                recent_f = self.flight_steps_list[-50:]
                frac = self.flight_ep_count / max(self.ep_count, 1)
                print(
                    f"  [Jump] step={self.num_timesteps:,}  "
                    f"flight_rate={frac:.1%}  "
                    f"avg_h={np.mean(recent_h):.3f}m  "
                    f"max_h={np.max(recent_h):.3f}m  "
                    f"avg_flight={np.mean(recent_f):.1f}steps"
                )
                self.logger.record("jump/avg_height", np.mean(recent_h))
                self.logger.record("jump/max_height", np.max(recent_h))
                self.logger.record("jump/flight_rate", frac)
            else:
                print(
                    f"  [Jump] step={self.num_timesteps:,}  "
                    f"NO FLIGHTS YET ({self.ep_count} episodes)"
                )
        return True


class DiagnosticCallback(BaseCallback):
    def __init__(self, print_every=100_000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            logs = self.logger.name_to_value
            kl = logs.get("train/approx_kl", "?")
            clip = logs.get("train/clip_fraction", "?")
            ev = logs.get("train/explained_variance", "?")
            std = logs.get("train/std", "?")
            print(
                f"  [Health] step={self.num_timesteps:,}  "
                f"kl={kl}  clip={clip}  expl_var={ev}  std={std}"
            )
        return True


# =============================================================================
# Environment factory
# =============================================================================
def make_env(max_episode_steps=500, seed=0):
    """Factory for SubprocVecEnv — must be picklable (top-level function)."""
    def _init():
        env = gym.make("Humanoid-v5", render_mode=None)
        env = JumpRewardWrapper(env, max_episode_steps=max_episode_steps)
        env.reset(seed=seed)
        return env
    return _init


# =============================================================================
# Training
# =============================================================================
def run_training(n_envs, total_timesteps, max_episode_steps):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Humanoid Jump Training v2d ===")
    print(f"  Parallel envs:     {n_envs}")
    print(f"  Total timesteps:   {total_timesteps:,}")
    print(f"  Max episode steps: {max_episode_steps}")
    print(f"  Device:            {device}")
    print()

    # ── Parallel environments ──
    venv = SubprocVecEnv([
        make_env(max_episode_steps=max_episode_steps, seed=i)
        for i in range(n_envs)
    ])
    env = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # ── PPO ──
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        device=device,
        policy_kwargs=dict(
            activation_fn=nn.ELU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        tensorboard_log="./jump_tb_logs/",
    )

    # ── Train ──
    print("--- Starting training ---\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            JumpMetricsCallback(print_every=20_000),
            DiagnosticCallback(print_every=100_000),
        ],
    )
    print("\n--- Training complete ---")

    model.save("humanoid_jumper_v2")
    env.save("humanoid_jumper_v2_vecnorm.pkl")
    env.close()
    print("Saved: humanoid_jumper_v2.zip + humanoid_jumper_v2_vecnorm.pkl")


# =============================================================================
# Entry point (required for SubprocVecEnv on Windows)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Humanoid Jump v2d")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--episode-steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    run_training(args.envs, args.timesteps, args.episode_steps)
