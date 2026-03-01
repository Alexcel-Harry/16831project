# =============================================================================
# backflip_train_ppo.py — PPO Baseline for Backflip (Windows/Linux/Mac)
# =============================================================================
#
# Usage:
#   python backflip_train_ppo.py
#   python backflip_train_ppo.py --timesteps 10000000 --envs 8
#
# Requirements:
#   pip install gymnasium[mujoco] stable-baselines3 torch
#
# Output:
#   backflip_ppo.zip           — trained PPO model
#   backflip_ppo_vecnorm.pkl   — observation/reward normalization stats
# =============================================================================

import argparse
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from backflip_env import BackflipRewardWrapper


# =============================================================================
# Callbacks
# =============================================================================
class BackflipMetricsCallback(BaseCallback):
    def __init__(self, print_every=20_000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self.heights = []
        self.rotations = []
        self.foot_zs = []
        self.ep_count = 0
        self.flight_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.ep_count += 1
            if "jump_height" in info:
                self.heights.append(info["jump_height"])
                self.flight_count += 1
            if "best_rotation_deg" in info:
                self.rotations.append(info["best_rotation_deg"])
            if "max_foot_z" in info:
                self.foot_zs.append(info["max_foot_z"])

        if self.num_timesteps % self.print_every == 0:
            frac = self.flight_count / max(self.ep_count, 1)
            if self.rotations:
                rr = self.rotations[-50:]
                rh = self.heights[-50:] if self.heights else [0]
                rf = self.foot_zs[-50:] if self.foot_zs else [0]
                print(
                    f"  [PPO] step={self.num_timesteps:,}  "
                    f"flight={frac:.1%}  "
                    f"avg_rot={np.mean(rr):.1f}°  max_rot={np.max(rr):.1f}°  "
                    f"avg_h={np.mean(rh):.3f}m  avg_foot_z={np.mean(rf):.2f}"
                )
                self.logger.record("backflip/avg_rotation", np.mean(rr))
                self.logger.record("backflip/max_rotation", np.max(rr))
                self.logger.record("backflip/avg_height", np.mean(rh))
                self.logger.record("backflip/flight_rate", frac)
            else:
                print(
                    f"  [PPO] step={self.num_timesteps:,}  "
                    f"NO FLIGHTS YET ({self.ep_count} eps)"
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
def make_env(max_episode_steps=500, seed=0,
             jump_scale=1.0, flip_scale=1.0):
    def _init():
        env = gym.make("Humanoid-v5", render_mode=None)
        env = BackflipRewardWrapper(
            env,
            max_episode_steps=max_episode_steps,
            jump_reward_scale=jump_scale,
            backflip_reward_scale=flip_scale,
        )
        env.reset(seed=seed)
        return env
    return _init


# =============================================================================
# Training
# =============================================================================
def run_training(n_envs, total_timesteps, max_episode_steps,
                 jump_scale, flip_scale):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Backflip PPO Baseline ===")
    print(f"  Parallel envs:     {n_envs}")
    print(f"  Total timesteps:   {total_timesteps:,}")
    print(f"  Max episode steps: {max_episode_steps}")
    print(f"  Device:            {device}")
    print(f"  Jump scale:        {jump_scale}")
    print(f"  Flip scale:        {flip_scale}")
    print()

    venv = SubprocVecEnv([
        make_env(max_episode_steps, seed=i,
                 jump_scale=jump_scale, flip_scale=flip_scale)
        for i in range(n_envs)
    ])
    env = VecNormalize(
        venv, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )

    model = PPO(
        "MlpPolicy", env, verbose=1,
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
        tensorboard_log="./backflip_ppo_tb/",
    )

    print("--- Starting PPO training ---\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            BackflipMetricsCallback(print_every=20_000),
            DiagnosticCallback(print_every=100_000),
        ],
    )
    print("\n--- Training complete ---")

    model.save("backflip_ppo")
    env.save("backflip_ppo_vecnorm.pkl")
    env.close()
    print("Saved: backflip_ppo.zip + backflip_ppo_vecnorm.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Backflip PPO Baseline")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--jump-scale", type=float, default=1.0)
    parser.add_argument("--flip-scale", type=float, default=1.0)
    args = parser.parse_args()

    run_training(args.envs, args.timesteps, args.episode_steps,
                 args.jump_scale, args.flip_scale)
