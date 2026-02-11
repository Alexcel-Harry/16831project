# =============================================================================
# plot_performance.py — Plot & summarize random agent results from log.csv
# =============================================================================
#
# Reads per-step log.csv and produces:
#   - Per-step reward over time
#   - Cumulative reward over time
#   - Summary statistics
#
# Usage:
#   python plot_performance.py
#   python plot_performance.py --csv my_log.csv
# =============================================================================

import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def main(csv_path="log.csv", out_png="random_agent_performance.png"):
    # ── Read CSV ──
    steps, episodes, rewards, cum_rewards, zs, vzs = [], [], [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            cum_rewards.append(float(row["cumulative_reward"]))
            zs.append(float(row["z"]))
            vzs.append(float(row["vz"]))

    steps = np.array(steps)
    rewards = np.array(rewards)
    cum_rewards = np.array(cum_rewards)
    zs = np.array(zs)

    # ── Print summary ──
    print(f"{'='*50}")
    print(f"  Random Agent Performance Summary")
    print(f"{'='*50}")
    print(f"  Total steps:        {len(steps)}")
    print(f"  Episodes:           {episodes[-1]}")
    print(f"  Mean step reward:   {rewards.mean():.4f} ± {rewards.std():.4f}")
    print(f"  Median step reward: {np.median(rewards):.4f}")
    print(f"  Min / Max reward:   {rewards.min():.4f} / {rewards.max():.4f}")
    print(f"  Final cumulative:   {cum_rewards[-1]:.2f}")
    print(f"  Mean z:             {zs.mean():.3f}")
    print(f"{'='*50}")

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Top: per-step reward
    ax1 = axes[0]
    ax1.plot(steps, rewards, color="steelblue", alpha=0.6, linewidth=0.8)
    ax1.axhline(rewards.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {rewards.mean():.4f}")
    ax1.set_ylabel("Step Reward")
    ax1.set_title("Random Agent — Per-Step Reward")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Middle: cumulative reward
    ax2 = axes[1]
    ax2.plot(steps, cum_rewards, color="darkorange", linewidth=1.2)
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Random Agent — Cumulative Reward Over Time")
    ax2.grid(axis="y", alpha=0.3)

    # Bottom: z height
    ax3 = axes[2]
    ax3.plot(steps, zs, color="seagreen", alpha=0.7, linewidth=0.8)
    ax3.axhline(zs.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean z = {zs.mean():.3f}")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("z (height)")
    ax3.set_title("Random Agent — Humanoid Height Over Time")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\n  Plot saved to: {os.path.abspath(out_png)}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="log.csv", help="Path to log CSV")
    parser.add_argument("--out", default="random_agent_performance.png", help="Output plot filename")
    args = parser.parse_args()
    main(csv_path=args.csv, out_png=args.out)