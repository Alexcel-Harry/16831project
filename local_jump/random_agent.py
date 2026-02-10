# =============================================================================
# random_agent.py — Random Agent Baseline (Standalone)
# =============================================================================
#
# Shows a random agent flailing around in the Humanoid-v5 environment
# with the jump reward wrapper, so you can compare against the trained agent.
#
# Usage:
#   python random_agent.py               # watch random agent with reward log
#   python random_agent.py --steps 1000  # run for 1000 steps
#   python random_agent.py --no-render   # headless, just print rewards
#
# This is equivalent to:  python jump_check.py --random
# but self-contained (no trained model files needed).
#
# Requirements:
#   pip install gymnasium[mujoco]
# =============================================================================

import argparse
import time
import numpy as np
from jump_env import JumpRewardWrapper, STANDING_Z
import gymnasium as gym


def run_random(max_steps=500, render=True, seed=42):
    render_mode = "human" if render else None
    raw_env = gym.make("Humanoid-v5", render_mode=render_mode)
    env = JumpRewardWrapper(raw_env, max_episode_steps=max_steps)

    print(f"{'='*70}")
    print(f"  RANDOM AGENT — Humanoid-v5 with Jump Reward Wrapper")
    print(f"  Steps: {max_steps}   Render: {'ON' if render else 'OFF'}")
    print(f"{'='*70}\n")

    obs, info = env.reset(seed=seed)
    cumulative = 0.0
    episode = 0
    max_z_seen = 0.0
    total_flight = 0

    header = (
        f"{'step':>5} │ {'z':>6} {'vz':>6} {'air':>3} │ "
        f"{'reward':>7} {'cumul':>8} │ {'notes'}"
    )
    print(header)
    print("─" * 70)

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward

        z = info.get("z", 0)
        vz = info.get("vz", 0)
        air = info.get("airborne", False)
        max_z_seen = max(max_z_seen, z)

        notes = ""
        if air:
            total_flight += 1
            notes = f"AIRBORNE (h={z - STANDING_Z:+.3f}m)"
        if info.get("reward_breakdown", {}).get("posture", 0) < 0:
            notes = "FELL"

        # Print every 10 steps, or on notable events
        if step % 10 == 0 or air or terminated:
            flag = "✓" if air else " "
            print(
                f"{step:>5} │ {z:>6.3f} {vz:>6.2f} {flag:>3} │ "
                f"{reward:>7.3f} {cumulative:>8.2f} │ {notes}"
            )

        if terminated or truncated:
            episode += 1
            reason = "terminated" if terminated else "truncated"
            print(f"\n  ── Episode {episode} ended ({reason}) at step {step} ──")
            print(f"     Cumulative reward: {cumulative:.2f}")
            print(f"     Max z this episode: {max_z_seen:.3f}\n")

            obs, info = env.reset()
            cumulative = 0.0
            max_z_seen = 0.0
            print(header)
            print("─" * 70)

        if render:
            time.sleep(0.01)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Total steps:   {max_steps}")
    print(f"  Episodes:      {episode}")
    print(f"  Flight steps:  {total_flight}  ({total_flight/max_steps*100:.1f}%)")
    print(f"{'='*70}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Agent Baseline")
    parser.add_argument("--steps", type=int, default=500, help="Steps to run")
    parser.add_argument("--no-render", action="store_true", help="No MuJoCo viewer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_random(max_steps=args.steps, render=not args.no_render, seed=args.seed)
