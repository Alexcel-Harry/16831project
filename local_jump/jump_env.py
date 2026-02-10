# =============================================================================
# jump_env.py — Humanoid Jump Reward Wrapper v2d
# =============================================================================
# Shared module imported by jump_train.py, jump_check.py, and random_agent.py.
# Contains the JumpRewardWrapper with contact-based flight detection,
# quadratic height scaling, crouch preparation, and liftoff velocity bonus.
# =============================================================================

import gymnasium as gym
import numpy as np

# ── Geom IDs (Humanoid-v5) ──────────────────────────────────────────────────
FLOOR_GEOM_ID      = 0
RIGHT_FOOT_GEOM_ID = 8
LEFT_FOOT_GEOM_ID  = 11
FOOT_GEOM_IDS      = {RIGHT_FOOT_GEOM_ID, LEFT_FOOT_GEOM_ID}

STANDING_Z = 1.4


class JumpRewardWrapper(gym.Wrapper):
    """
    Phase-aware jump reward with contact-based flight detection.

    Reward gradient (approximate per-step):
        standing still    →  ~0.08   (just height baseline)
        crouching (load)  →  ~0.3    (crouch reward)
        extending (vz=1)  →  ~0.6    (velocity reward)
        micro-hop (z=1.2) →  ~1.5    (flight base)
        hop at z=1.5      →  ~2.0    (flight + small height²)
        real jump z=1.8   →  ~4.6    (flight + large height²)
        great jump z=2.2  →  ~9.4    (flight + huge height²)
    """

    def __init__(self, env, max_episode_steps=500):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._reset_tracking()

    def _reset_tracking(self):
        self.step_count = 0
        self.max_z = -np.inf
        self.total_flight_steps = 0
        self.max_flight_z = -np.inf
        self.entered_flight = False

        # Crouch tracking
        self.was_crouching = False
        self.min_z_this_attempt = STANDING_Z

        # Liftoff tracking
        self.was_grounded = True
        self.vz_at_liftoff = 0.0
        self.current_jump_steps = 0

        # Action history for jerk penalty
        self.action_hist = [
            np.zeros(self.action_space.shape),
            np.zeros(self.action_space.shape),
        ]

        # Per-step reward breakdown (for logging)
        self.last_reward_breakdown = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_tracking()
        return obs, info

    # ── Contact helpers ──────────────────────────────────────────────────
    def _get_floor_contacts(self):
        d = self.unwrapped.data
        touching_floor = set()
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if FLOOR_GEOM_ID in pair:
                other = (pair - {FLOOR_GEOM_ID}).pop()
                touching_floor.add(other)
        return touching_floor

    def _feet_on_ground(self, floor_contacts):
        return len(FOOT_GEOM_IDS & floor_contacts)

    def _non_foot_on_ground(self, floor_contacts):
        return len(floor_contacts - FOOT_GEOM_IDS) > 0

    # ── Main step ────────────────────────────────────────────────────────
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── Gather state ──
        x, y, z = self.unwrapped.data.qpos[0:3]
        vx, vy, vz = self.unwrapped.data.qvel[0:3]
        floor_contacts = self._get_floor_contacts()
        n_feet = self._feet_on_ground(floor_contacts)
        body_fell = self._non_foot_on_ground(floor_contacts)
        airborne = (n_feet == 0) and (not body_fell)

        self.max_z = max(self.max_z, z)

        # ── Track crouch & liftoff ──
        grounded = (n_feet > 0)
        if grounded:
            self.min_z_this_attempt = min(self.min_z_this_attempt, z)
            if z < STANDING_Z - 0.1:
                self.was_crouching = True
            if not self.was_grounded:
                self.current_jump_steps = 0
                self.was_crouching = False
                self.min_z_this_attempt = z

        if self.was_grounded and not grounded and not body_fell:
            self.vz_at_liftoff = max(vz, 0.0)
            self.current_jump_steps = 0
        self.was_grounded = grounded

        # ══════════════════════════════════════════════════════════════════
        # REWARD
        # ══════════════════════════════════════════════════════════════════
        reward = 0.0
        bd = {}  # breakdown for logging

        # ▸ 1. UPWARD VELOCITY
        r_vel = 0.5 * np.clip(vz, 0.0, 5.0)
        reward += r_vel
        bd["velocity"] = r_vel

        # ▸ 2. FLIGHT BONUS — linear + quadratic height
        r_flight = 0.0
        if airborne and z > 0.8:
            height_above = max(0.0, z - STANDING_Z)
            r_flight += 1.0
            r_flight += 3.0 * height_above
            r_flight += 8.0 * height_above ** 2
            r_flight += 1.0 * np.clip(vz, 0.0, 5.0)

            self.current_jump_steps += 1
            self.total_flight_steps += 1
            self.max_flight_z = max(self.max_flight_z, z)
            self.entered_flight = True
        reward += r_flight
        bd["flight"] = r_flight

        # ▸ 3. LIFTOFF VELOCITY BONUS
        r_liftoff = 0.0
        if airborne and z > 0.8 and self.current_jump_steps == 1:
            r_liftoff = 0.2 * self.vz_at_liftoff ** 2
        reward += r_liftoff
        bd["liftoff"] = r_liftoff

        # ▸ 4. CROUCH PREPARATION
        r_crouch = 0.0
        if grounded and z < STANDING_Z - 0.05 and z > 0.8:
            crouch_depth = STANDING_Z - z
            r_crouch = 0.3 * min(crouch_depth, 0.3)
        reward += r_crouch
        bd["crouch"] = r_crouch

        # ▸ 5. HEIGHT BASELINE
        r_height = 0.2 * max(0.0, z - 1.0)
        reward += r_height
        bd["height"] = r_height

        # ▸ 6. POSTURE
        r_posture = -3.0 if body_fell else 0.0
        reward += r_posture
        bd["posture"] = r_posture

        # ▸ 7. STAY CENTERED
        drift = np.sqrt(x**2 + y**2)
        r_drift = -0.1 * drift
        reward += r_drift
        bd["drift"] = r_drift

        # ▸ 8. JERK PENALTY
        jerk = action - 2.0 * self.action_hist[-1] + self.action_hist[-2]
        r_jerk = -0.02 * np.sum(np.square(jerk))
        reward += r_jerk
        bd["jerk"] = r_jerk

        # ▸ 9. CONTROL COST
        r_ctrl = -0.005 * np.sum(np.square(action))
        reward += r_ctrl
        bd["control"] = r_ctrl

        bd["total"] = reward
        self.last_reward_breakdown = bd

        # ▸ 10. EPISODE END
        is_done = terminated or truncated or (self.step_count >= self.max_episode_steps)
        if is_done:
            if self.entered_flight:
                info["jump_height"] = self.max_flight_z - STANDING_Z
                info["flight_steps"] = self.total_flight_steps
            if not terminated and not truncated:
                truncated = True

        # ── TERMINATION ──
        if z < 0.3:
            terminated = True
            reward -= 3.0
        if drift > 1.0:
            terminated = True
            reward -= 2.0

        # Update action history
        self.action_hist[-2] = self.action_hist[-1].copy()
        self.action_hist[-1] = action.copy()

        # Extra info for viewers
        info["z"] = z
        info["vz"] = vz
        info["airborne"] = airborne
        info["drift"] = drift
        info["reward_breakdown"] = bd

        return obs, reward, terminated, truncated, info
