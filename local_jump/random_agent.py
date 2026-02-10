# =============================================================================
# random_agent.py — Random Agent Baseline (with Video Recording)
# =============================================================================
#
# Shows a random agent flailing around in the Humanoid-v5 environment.
# Now includes options to record video for review.
#
# Usage:
#   python random_agent.py               # Live view (Human mode)
#   python random_agent.py --record      # Save video to ./videos/random (No live view)
#   python random_agent.py --steps 1000  # Run for longer
#
# Requirements:
#   pip install gymnasium[mujoco] moviepy
# =============================================================================

import argparse
import time
import os
import gymnasium as gym
from jump_env import JumpRewardWrapper, STANDING_Z

def run_random(max_steps=500, render=True, record=False, seed=42):
    # ── 1. Configure Render Mode ──
    # If recording, we MUST use "rgb_array".
    # If not recording but rendering, we use "human" (live window).
    if record:
        render_mode = "rgb_array"
        print(f"\n[INFO] Video recording enabled. Mode set to 'rgb_array'.")
        print(f"[INFO] No live window will appear. Check ./videos/random after run.\n")
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # ── 2. Create Environment ──
    raw_env = gym.make("Humanoid-v5", render_mode=render_mode)

    # ── 3. Attach Video Recorder (Optional) ──
    if record:
        # Save videos to ./videos/random
        # name_prefix helps identify the file
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=os.path.join("videos", "random"),
            episode_trigger=lambda episode_id: True,  # Record every episode
            name_prefix="random-humanoid",
            disable_logger=False
        )

    # ── 4. Attach Jump Wrapper ──
    # We wrap the env (which might be the recorder) with our logic
    env = JumpRewardWrapper(raw_env, max_episode_steps=max_steps)

    print(f"{'='*70}")
    print(f"  RANDOM AGENT — Humanoid-v5")
    print(f"  Steps: {max_steps}   Recording: {'ON' if record else 'OFF'}")
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
        # Sample random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward

        # Metrics
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

        # Logging
        if step % 10 == 0 or air or terminated:
            flag = "✓" if air else " "
            print(
                f"{step:>5} │ {z:>6.3f} {vz:>6.2f} {flag:>3} │ "
                f"{reward:>7.3f} {cumulative:>8.2f} │ {notes}"
            )

        # Handle Episode End
        if terminated or truncated:
            episode += 1
            reason = "terminated" if terminated else "truncated"
            print(f"\n  ── Episode {episode} ended ({reason}) at step {step} ──")
            print(f"     Cumulative reward: {cumulative:.2f}")
            print(f"     Max z this episode: {max_z_seen:.3f}\n")

            # Environment resets automatically inside wrappers usually, 
            # but we call reset() to ensure clean state and get new obs.
            obs, info = env.reset()
            cumulative = 0.0
            max_z_seen = 0.0
            print(header)
            print("─" * 70)

        # Slow down ONLY if watching live. 
        # If recording, we want it to run as fast as possible.
        if render and not record:
            time.sleep(0.01)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Total steps:   {max_steps}")
    print(f"  Episodes:      {episode}")
    print(f"  Flight steps:  {total_flight}")
    if record:
        print(f"  Video saved to: {os.path.abspath(os.path.join('videos', 'random'))}")
    print(f"{'='*70}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Agent Baseline")
    parser.add_argument("--steps", type=int, default=500, help="Steps to run")
    parser.add_argument("--no-render", action="store_true", help="No MuJoCo viewer (headless)")
    parser.add_argument("--record", action="store_true", help="Record video (saves to ./videos/random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Logic: If --record is passed, we effectively want to render frames for the video, 
    # even if --no-render was accidentally passed.
    render_active = (not args.no_render) or args.record

    run_random(
        max_steps=args.steps, 
        render=render_active, 
        record=args.record, 
        seed=args.seed
    )