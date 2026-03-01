# =============================================================================
# backflip_train_dqn.py — DQN Baseline for Backflip (Windows/Linux/Mac)
# =============================================================================
#
# DQN requires discrete actions.  The Humanoid has 17 continuous action dims,
# so we discretise via a fixed codebook of action vectors (structured
# primitives + random samples clustered with k-means-style init).
#
# NOTE: DQN on high-dim continuous control is inherently limited compared
# to PPO.  This is included as a baseline for comparison.
#
# Usage:
#   python backflip_train_dqn.py
#   python backflip_train_dqn.py --timesteps 5000000 --n-actions 128
#
# Output:
#   backflip_dqn.zip           — trained DQN model
#   backflip_dqn_vecnorm.pkl   — observation/reward normalization stats
# =============================================================================

import argparse
import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from backflip_env import BackflipRewardWrapper


# =============================================================================
# Discrete action wrapper
# =============================================================================
class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Maps Discrete(n_actions) → continuous action space using a fixed codebook.

    Codebook = structured primitives (neutral, extend, flex, per-joint)
             + random samples to fill the remaining slots.
    """

    def __init__(self, env, n_actions=64, seed=42):
        super().__init__(env)
        self.n_actions = n_actions
        self.continuous_space = env.action_space
        self.action_space = spaces.Discrete(n_actions)

        rng = np.random.RandomState(seed)
        low = self.continuous_space.low
        high = self.continuous_space.high
        dim = self.continuous_space.shape[0]

        # Structured primitives
        primitives = [
            np.zeros(dim),                  # neutral
            np.ones(dim) * 0.5,             # moderate extend
            np.ones(dim) * -0.5,            # moderate flex
            np.ones(dim),                   # max extend
            -np.ones(dim),                  # max flex
        ]
        # Per-joint on/off (both directions)
        for i in range(min(dim, 17)):
            v_pos = np.zeros(dim); v_pos[i] = 1.0
            v_neg = np.zeros(dim); v_neg[i] = -1.0
            primitives.extend([v_pos, v_neg])

        # Fill remaining with random
        n_random = max(0, n_actions - len(primitives))
        random_acts = rng.uniform(low, high, size=(n_random, dim))
        all_acts = primitives[:n_actions]
        if len(all_acts) < n_actions:
            all_acts = primitives + list(random_acts)

        self.codebook = np.clip(
            np.array(all_acts[:n_actions], dtype=np.float32), low, high
        )

    def action(self, act_idx):
        return self.codebook[act_idx]


# =============================================================================
# Callbacks
# =============================================================================
class DQNBackflipCallback(BaseCallback):
    def __init__(self, print_every=20_000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self.heights = []
        self.rotations = []
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

        if self.num_timesteps % self.print_every == 0:
            frac = self.flight_count / max(self.ep_count, 1)
            if self.rotations:
                rr = self.rotations[-50:]
                rh = self.heights[-50:] if self.heights else [0]
                print(
                    f"  [DQN] step={self.num_timesteps:,}  "
                    f"flight={frac:.1%}  "
                    f"avg_rot={np.mean(rr):.1f}°  max_rot={np.max(rr):.1f}°  "
                    f"avg_h={np.mean(rh):.3f}m  eps={self.ep_count}"
                )
            else:
                print(
                    f"  [DQN] step={self.num_timesteps:,}  "
                    f"NO FLIGHTS ({self.ep_count} eps)"
                )
        return True


# =============================================================================
# Environment factory
# =============================================================================
def make_env(max_episode_steps=500, n_actions=64, seed=0,
             jump_scale=1.0, flip_scale=1.0):
    def _init():
        env = gym.make("Humanoid-v5", render_mode=None)
        env = BackflipRewardWrapper(
            env,
            max_episode_steps=max_episode_steps,
            jump_reward_scale=jump_scale,
            backflip_reward_scale=flip_scale,
        )
        env = DiscreteActionWrapper(env, n_actions=n_actions, seed=seed)
        env.reset(seed=seed)
        return env
    return _init


# =============================================================================
# Training
# =============================================================================
def run_training(total_timesteps, max_episode_steps, n_actions,
                 jump_scale, flip_scale):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Backflip DQN Baseline ===")
    print(f"  Discrete actions:  {n_actions}")
    print(f"  Total timesteps:   {total_timesteps:,}")
    print(f"  Max episode steps: {max_episode_steps}")
    print(f"  Device:            {device}")
    print(f"  Jump scale:        {jump_scale}")
    print(f"  Flip scale:        {flip_scale}")
    print()

    # DQN uses single env (DummyVecEnv)
    venv = DummyVecEnv([
        make_env(max_episode_steps, n_actions, seed=0,
                 jump_scale=jump_scale, flip_scale=flip_scale)
    ])
    env = VecNormalize(
        venv, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )

    model = DQN(
        "MlpPolicy", env, verbose=1,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./backflip_dqn_tb/",
    )

    print(f"--- Starting DQN training ({n_actions} discrete actions) ---\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[DQNBackflipCallback(print_every=20_000)],
    )
    print("\n--- Training complete ---")

    model.save("backflip_dqn")
    env.save("backflip_dqn_vecnorm.pkl")
    env.close()
    print("Saved: backflip_dqn.zip + backflip_dqn_vecnorm.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Backflip DQN Baseline")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--n-actions", type=int, default=64)
    parser.add_argument("--jump-scale", type=float, default=1.0)
    parser.add_argument("--flip-scale", type=float, default=1.0)
    args = parser.parse_args()

    run_training(args.timesteps, args.episode_steps, args.n_actions,
                 args.jump_scale, args.flip_scale)
