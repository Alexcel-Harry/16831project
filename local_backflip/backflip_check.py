# =============================================================================
# backflip_check.py — Visualize Backflip Agent (PPO or DQN)
# =============================================================================
#
# Usage:
#   python backflip_check.py --algo ppo                    # PPO trained model
#   python backflip_check.py --algo dqn                    # DQN trained model
#   python backflip_check.py --algo ppo --random           # random baseline
#   python backflip_check.py --algo ppo --no-render        # headless
#   python backflip_check.py --algo dqn --log out.csv      # save CSV
#
# Requirements:
#   pip install gymnasium[mujoco] stable-baselines3
# =============================================================================

import argparse
import os
import sys
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backflip_env import BackflipRewardWrapper, STANDING_Z


# ── Discrete wrapper (must match training) ───────────────────────────────────
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions=64, seed=42):
        super().__init__(env)
        self.n_actions = n_actions
        self.continuous_space = env.action_space
        self.action_space = spaces.Discrete(n_actions)
        rng = np.random.RandomState(seed)
        low, high = self.continuous_space.low, self.continuous_space.high
        dim = self.continuous_space.shape[0]
        primitives = [
            np.zeros(dim), np.ones(dim)*0.5, -np.ones(dim)*0.5,
            np.ones(dim), -np.ones(dim),
        ]
        for i in range(min(dim, 17)):
            v_pos = np.zeros(dim); v_pos[i] = 1.0
            v_neg = np.zeros(dim); v_neg[i] = -1.0
            primitives.extend([v_pos, v_neg])
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


# ── Logging helpers ──────────────────────────────────────────────────────────
def print_header():
    print(
        f"{'step':>6} │ {'z':>6} {'vz':>6} {'air':>4} {'pitch°':>7} "
        f"{'footZ':>6} │ {'vel':>6} {'flight':>7} {'angV':>5} "
        f"{'feetH':>5} {'rot':>5} {'land':>5} │ "
        f"{'TOTAL':>7} {'cum':>8}"
    )
    print("─" * 105)


def print_step(step, info, reward, cumulative):
    bd = info.get("reward_breakdown", {})
    z = info.get("z", 0)
    vz = info.get("vz", 0)
    air = "✓" if info.get("airborne", False) else ""
    pitch_deg = info.get("cumulative_pitch_deg", 0)
    foot_z = info.get("foot_max_z", 0)

    print(
        f"{step:>6} │ "
        f"{z:>6.3f} {vz:>6.2f} {air:>4} {pitch_deg:>7.1f} {foot_z:>6.3f} │ "
        f"{bd.get('velocity', 0):>6.3f} "
        f"{bd.get('flight', 0):>7.3f} "
        f"{bd.get('flip_ang_velocity', 0):>5.2f} "
        f"{bd.get('flip_foot_height', 0):>5.2f} "
        f"{bd.get('flip_rotation', 0):>5.2f} "
        f"{bd.get('flip_landing', 0):>5.1f} │ "
        f"{reward:>7.3f} {cumulative:>8.2f}"
    )


# ── Main check function ─────────────────────────────────────────────────────
def run_check(
    algo: str = "ppo",
    random_agent: bool = False,
    max_steps: int = 500,
    render: bool = True,
    log_file: str = None,
    log_every: int = 1,
    n_actions: int = 64,
    jump_scale: float = 1.0,
    flip_scale: float = 1.0,
):
    is_dqn = (algo == "dqn")
    model_path = f"backflip_{algo}"
    vecnorm_path = f"backflip_{algo}_vecnorm.pkl"

    if not random_agent and not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}.zip")
        print(f"Train first: python backflip_train_{algo}.py")
        sys.exit(1)

    # ── Create env ──
    render_mode = "human" if render else None
    raw = gym.make("Humanoid-v5", render_mode=render_mode)
    env = BackflipRewardWrapper(raw, max_episode_steps=max_steps,
                                jump_reward_scale=jump_scale,
                                backflip_reward_scale=flip_scale)
    if is_dqn:
        env = DiscreteActionWrapper(env, n_actions=n_actions)

    # ── Load model ──
    model = None
    vec_env = None
    if not random_agent:
        def _make_dummy():
            e = gym.make("Humanoid-v5", render_mode=None)
            e = BackflipRewardWrapper(e, max_episode_steps=max_steps,
                                      jump_reward_scale=jump_scale,
                                      backflip_reward_scale=flip_scale)
            if is_dqn:
                e = DiscreteActionWrapper(e, n_actions=n_actions)
            return e
        dummy = DummyVecEnv([_make_dummy])
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, dummy)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"Loaded VecNormalize from {vecnorm_path}")
        else:
            print(f"WARNING: {vecnorm_path} not found, using raw obs.")

        loader = DQN if is_dqn else PPO
        model = loader.load(model_path)
        print(f"Loaded {algo.upper()} model from {model_path}.zip")

    # ── CSV log ──
    csv_file = None
    if log_file:
        csv_file = open(log_file, "w")
        csv_file.write(
            "step,z,vz,airborne,pitch_deg,foot_max_z,drift,"
            "r_velocity,r_flight,r_flip_ang_vel,r_flip_foot_h,"
            "r_flip_rotation,r_flip_landing,r_total,cumulative\n"
        )

    mode = "RANDOM" if random_agent else algo.upper()
    print(f"\n{'='*60}")
    print(f"  Backflip Check: {mode}")
    print(f"  Steps: {max_steps}   Render: {'ON' if render else 'OFF'}")
    print(f"{'='*60}\n")

    obs, info = env.reset()
    cum = 0.0
    ep_count = 0

    print_header()

    for step in range(max_steps):
        if random_agent:
            action = env.action_space.sample()
        else:
            obs_n = vec_env.normalize_obs(obs) if vec_env else obs
            action, _ = model.predict(obs_n, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        cum += reward

        if step % log_every == 0:
            print_step(step, info, reward, cum)

        if csv_file:
            bd = info.get("reward_breakdown", {})
            air = 1 if info.get("airborne", False) else 0
            csv_file.write(
                f"{step},{info.get('z',0):.4f},{info.get('vz',0):.4f},"
                f"{air},{info.get('cumulative_pitch_deg',0):.2f},"
                f"{info.get('foot_max_z',0):.4f},{info.get('drift',0):.4f},"
                f"{bd.get('velocity',0):.4f},{bd.get('flight',0):.4f},"
                f"{bd.get('flip_ang_velocity',0):.4f},"
                f"{bd.get('flip_foot_height',0):.4f},"
                f"{bd.get('flip_rotation',0):.4f},"
                f"{bd.get('flip_landing',0):.4f},"
                f"{reward:.4f},{cum:.4f}\n"
            )

        if terminated or truncated:
            ep_count += 1
            rot = info.get("best_rotation_deg", 0)
            print(
                f"\n  *** Episode {ep_count} ended "
                f"({'term' if terminated else 'trunc'}) "
                f"step {step}  cum={cum:.2f}  "
                f"rot={rot:.1f}° ***\n"
            )
            obs, info = env.reset()
            cum = 0.0
            print_header()

        if render:
            time.sleep(0.01)

    print(f"\n{'='*60}")
    print(f"  Finished {max_steps} steps, {ep_count} episodes")
    print(f"{'='*60}")

    if csv_file:
        csv_file.close()
        print(f"  Log: {log_file}")

    env.close()
    if vec_env is not None:
        vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Backflip Agent")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--n-actions", type=int, default=64)
    parser.add_argument("--jump-scale", type=float, default=1.0)
    parser.add_argument("--flip-scale", type=float, default=1.0)
    args = parser.parse_args()

    run_check(
        algo=args.algo,
        random_agent=args.random,
        max_steps=args.steps,
        render=not args.no_render,
        log_file=args.log,
        log_every=args.log_every,
        n_actions=args.n_actions,
        jump_scale=args.jump_scale,
        flip_scale=args.flip_scale,
    )
