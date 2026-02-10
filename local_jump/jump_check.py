# =============================================================================
# jump_check.py — Visualize Trained or Random Agent with Reward Log
# =============================================================================
#
# Usage:
#   python jump_check.py                        # run trained agent
#   python jump_check.py --random               # run random agent
#   python jump_check.py --steps 1000           # run for 1000 steps
#   python jump_check.py --no-render            # headless, log only
#   python jump_check.py --random --log out.csv # save reward log to CSV
#
# Requirements:
#   pip install gymnasium[mujoco] stable-baselines3
#   (trained model files: humanoid_jumper_v2.zip + humanoid_jumper_v2_vecnorm.pkl)
# =============================================================================

import argparse
import os
import sys
import time
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from jump_env import JumpRewardWrapper, STANDING_Z


def print_header():
    print(
        f"{'step':>6} │ {'z':>6} {'vz':>6} {'air':>4} {'drift':>5} │ "
        f"{'vel':>6} {'flight':>7} {'lftoff':>6} {'crouch':>6} "
        f"{'height':>6} {'post':>5} {'jerk':>6} {'ctrl':>6} │ "
        f"{'TOTAL':>7} {'cum':>8}"
    )
    print("─" * 115)


def print_step(step, info, reward, cumulative):
    bd = info.get("reward_breakdown", {})
    z = info.get("z", 0)
    vz = info.get("vz", 0)
    air = "✓" if info.get("airborne", False) else ""
    drift = info.get("drift", 0)

    print(
        f"{step:>6} │ "
        f"{z:>6.3f} {vz:>6.2f} {air:>4} {drift:>5.3f} │ "
        f"{bd.get('velocity', 0):>6.3f} "
        f"{bd.get('flight', 0):>7.3f} "
        f"{bd.get('liftoff', 0):>6.3f} "
        f"{bd.get('crouch', 0):>6.3f} "
        f"{bd.get('height', 0):>6.3f} "
        f"{bd.get('posture', 0):>5.1f} "
        f"{bd.get('jerk', 0):>6.3f} "
        f"{bd.get('control', 0):>6.3f} │ "
        f"{reward:>7.3f} {cumulative:>8.2f}"
    )


def run_check(
    random_agent: bool = False,
    model_path: str = "humanoid_jumper_v2",
    vecnorm_path: str = "humanoid_jumper_v2_vecnorm.pkl",
    max_steps: int = 500,
    render: bool = True,
    log_file: str = None,
    log_every: int = 1,
):
    # ── Validate files ──
    if not random_agent:
        if not os.path.exists(model_path + ".zip"):
            print(f"ERROR: Model not found at {model_path}.zip")
            print("Train first with: python jump_train.py")
            print("Or run with --random for a random agent.")
            sys.exit(1)

    # ── Create environment ──
    render_mode = "human" if render else None
    raw_env = gym.make("Humanoid-v5", render_mode=render_mode)
    env = JumpRewardWrapper(raw_env, max_episode_steps=max_steps)

    # ── Load model ──
    model = None
    vec_env = None
    if not random_agent:
        # For a trained model, we need VecNormalize to transform observations
        # the same way they were transformed during training.
        vec_env_wrapped = DummyVecEnv([lambda: JumpRewardWrapper(
            gym.make("Humanoid-v5", render_mode=None),
            max_episode_steps=max_steps,
        )])
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env_wrapped)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"Loaded VecNormalize stats from {vecnorm_path}")
        else:
            print(f"WARNING: {vecnorm_path} not found, using raw observations.")
            vec_env = None

        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}.zip")

    # ── CSV log ──
    csv_file = None
    if log_file:
        csv_file = open(log_file, "w")
        csv_file.write(
            "step,z,vz,airborne,drift,"
            "r_velocity,r_flight,r_liftoff,r_crouch,r_height,"
            "r_posture,r_jerk,r_control,r_total,cumulative\n"
        )

    # ── Run ──
    mode = "RANDOM AGENT" if random_agent else "TRAINED AGENT"
    print(f"\n{'='*60}")
    print(f"  Running: {mode}")
    print(f"  Steps:   {max_steps}")
    print(f"  Render:  {'ON' if render else 'OFF'}")
    if log_file:
        print(f"  Log:     {log_file}")
    print(f"{'='*60}\n")

    obs, info = env.reset()
    cumulative_reward = 0.0
    episode_count = 0
    flight_count = 0

    print_header()

    for step in range(max_steps):
        # ── Choose action ──
        if random_agent:
            action = env.action_space.sample()
        else:
            # Normalize obs the same way training did
            if vec_env is not None:
                obs_norm = vec_env.normalize_obs(obs)
            else:
                obs_norm = obs
            action, _ = model.predict(obs_norm, deterministic=True)

        # ── Step ──
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        # ── Log ──
        if step % log_every == 0:
            print_step(step, info, reward, cumulative_reward)

        if csv_file:
            bd = info.get("reward_breakdown", {})
            air = 1 if info.get("airborne", False) else 0
            csv_file.write(
                f"{step},{info.get('z',0):.4f},{info.get('vz',0):.4f},"
                f"{air},{info.get('drift',0):.4f},"
                f"{bd.get('velocity',0):.4f},{bd.get('flight',0):.4f},"
                f"{bd.get('liftoff',0):.4f},{bd.get('crouch',0):.4f},"
                f"{bd.get('height',0):.4f},{bd.get('posture',0):.4f},"
                f"{bd.get('jerk',0):.4f},{bd.get('control',0):.4f},"
                f"{reward:.4f},{cumulative_reward:.4f}\n"
            )

        # ── Episode boundary ──
        if terminated or truncated:
            episode_count += 1
            if info.get("airborne", False) or "jump_height" in info:
                flight_count += 1

            print(f"\n  *** Episode {episode_count} ended "
                  f"({'terminated' if terminated else 'truncated'}) "
                  f"at step {step}  cumulative={cumulative_reward:.2f} ***\n")

            obs, info = env.reset()
            cumulative_reward = 0.0
            print_header()

        if render:
            time.sleep(0.01)  # slow down for human viewing

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Finished {max_steps} steps, {episode_count} episodes")
    print(f"{'='*60}")

    if csv_file:
        csv_file.close()
        print(f"  Log saved to: {log_file}")

    env.close()
    if vec_env is not None:
        vec_env.close()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Humanoid Jump Agent")
    parser.add_argument("--random", action="store_true", help="Use random agent instead of trained model")
    parser.add_argument("--model", type=str, default="humanoid_jumper_v2", help="Path to model (without .zip)")
    parser.add_argument("--vecnorm", type=str, default="humanoid_jumper_v2_vecnorm.pkl", help="Path to VecNormalize stats")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument("--no-render", action="store_true", help="Disable MuJoCo viewer (log only)")
    parser.add_argument("--log", type=str, default=None, help="Save reward log to CSV file")
    parser.add_argument("--log-every", type=int, default=1, help="Print log every N steps (default: every step)")
    args = parser.parse_args()

    run_check(
        random_agent=args.random,
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        max_steps=args.steps,
        render=not args.no_render,
        log_file=args.log,
        log_every=args.log_every,
    )
