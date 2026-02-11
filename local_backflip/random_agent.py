import argparse
import csv
import time
import os
import gymnasium as gym
from backflip_env import BackflipRewardWrapper, STANDING_Z

LOG_FILE = "log.csv"
LOG_FIELDS = ["step", "episode", "reward", "cumulative_reward", "z", "vz", "airborne"]

def run_random(max_steps=500, render=True, record=False, seed=42):
    
    if record:
        render_mode = "rgb_array"
        print(f"\n[INFO] Video recording enabled. Mode set to 'rgb_array'.")
        print(f"[INFO] No live window will appear. Check ./videos/random after run.\n")
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    raw_env = gym.make("Humanoid-v5", render_mode=render_mode, terminate_when_unhealthy=False)

    if record:
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=os.path.join("videos", "random"),
            episode_trigger=lambda episode_id: True,
            name_prefix="random-humanoid",
            disable_logger=False
        )

    env = raw_env

    print(f"{'='*70}")
    print(f"  RANDOM AGENT — Humanoid-v5")
    print(f"  Steps: {max_steps}   Recording: {'ON' if record else 'OFF'}")
    print(f"{'='*70}\n")

    csv_file = open(LOG_FILE, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    writer.writeheader()

    obs, info = env.reset(seed=seed)
    cumulative = 0.0
    episode = 1
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

        writer.writerow({
            "step":              step,
            "episode":           episode,
            "reward":            round(reward, 4),
            "cumulative_reward": round(cumulative, 4),
            "z":                 round(z, 4),
            "vz":                round(vz, 4),
            "airborne":          int(air),
        })

        if step % 10 == 0 or air or terminated:
            flag = "✓" if air else " "
            print(
                f"{step:>5} │ {z:>6.3f} {vz:>6.2f} {flag:>3} │ "
                f"{reward:>7.3f} {cumulative:>8.2f} │ {notes}"
            )

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\n  ── Episode {episode} ended ({reason}) at step {step} ──")
            print(f"     Cumulative reward: {cumulative:.2f}")
            print(f"     Max z this episode: {max_z_seen:.3f}\n")

            csv_file.flush()

            obs, info = env.reset()
            cumulative = 0.0
            max_z_seen = 0.0
            total_flight = 0
            episode += 1
            print(header)
            print("─" * 70)

        if render and not record:
            time.sleep(0.01)

    csv_file.close()

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Total steps:   {max_steps}")
    print(f"  Episodes:      {episode}")
    if record:
        print(f"  Video saved to: {os.path.abspath(os.path.join('videos', 'random'))}")
    print(f"  CSV log:       {os.path.abspath(LOG_FILE)}")
    print(f"{'='*70}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Agent Baseline")
    parser.add_argument("--steps", type=int, default=500, help="Steps to run")
    parser.add_argument("--no-render", action="store_true", help="No MuJoCo viewer (headless)")
    parser.add_argument("--record", action="store_true", help="Record video (saves to ./videos/random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    render_active = (not args.no_render) or args.record

    run_random(
        max_steps=args.steps,
        render=render_active,
        record=args.record,
        seed=args.seed
    )