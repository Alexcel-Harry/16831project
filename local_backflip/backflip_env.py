# =============================================================================
# backflip_env.py — Backflip Reward Wrapper (jump + rotation + foot height)
# =============================================================================
# Builds on top of the jump reward design. Reward is split into two composable
# functions so that each part can be modified independently:
#
#   compute_jump_rewards()     — the original jump shaping (velocity, flight,
#                                crouch, liftoff, height, posture, drift,
#                                jerk, control)
#   compute_backflip_rewards() — backflip-specific (rotation progress,
#                                angular velocity, foot height, tuck,
#                                landing quality, off-axis penalty)
#
# BackflipRewardWrapper = jump rewards + backflip rewards, each with its
# own scale factor for easy tuning.
# =============================================================================

import gymnasium as gym
import numpy as np

# ── Geom / body IDs (Humanoid-v5) ───────────────────────────────────────────
FLOOR_GEOM_ID      = 0
RIGHT_FOOT_GEOM_ID = 8
LEFT_FOOT_GEOM_ID  = 11
FOOT_GEOM_IDS      = {RIGHT_FOOT_GEOM_ID, LEFT_FOOT_GEOM_ID}

RIGHT_FOOT_BODY = "foot_right"
LEFT_FOOT_BODY  = "foot_left"

STANDING_Z = 1.4


# =============================================================================
# Quaternion helpers
# =============================================================================
def quat_to_euler(q):
    """Quaternion [w, x, y, z] → (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    roll  = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw   = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quat_to_pitch(q):
    """Quaternion [w, x, y, z] → pitch angle (rad)."""
    w, x, y, z = q
    return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))


# =============================================================================
# JUMP REWARDS  (composable function — identical to jump_env.py logic)
# =============================================================================
def compute_jump_rewards(z, vz, airborne, grounded, body_fell, x, y,
                         action, action_hist, current_jump_steps,
                         vz_at_liftoff, min_z_this_attempt):
    """
    Original jump-shaping rewards.  Returns (total, breakdown_dict).
    Kept as a standalone function so it can be called by both the
    JumpRewardWrapper and the BackflipRewardWrapper.
    """
    bd = {}
    reward = 0.0

    # 1. Upward velocity
    r_vel = 0.5 * np.clip(vz, 0.0, 5.0)
    reward += r_vel;  bd["velocity"] = r_vel

    # 2. Flight bonus — linear + quadratic height
    r_flight = 0.0
    if airborne and z > 0.8:
        ha = max(0.0, z - STANDING_Z)
        r_flight = 1.0 + 3.0 * ha + 8.0 * ha ** 2 + np.clip(vz, 0.0, 5.0)
    reward += r_flight;  bd["flight"] = r_flight

    # 3. Liftoff velocity bonus
    r_liftoff = 0.0
    if airborne and z > 0.8 and current_jump_steps == 1:
        r_liftoff = 0.2 * vz_at_liftoff ** 2
    reward += r_liftoff;  bd["liftoff"] = r_liftoff

    # 4. Crouch preparation
    r_crouch = 0.0
    if grounded and z < STANDING_Z - 0.05 and z > 0.8:
        r_crouch = 0.3 * min(STANDING_Z - z, 0.3)
    reward += r_crouch;  bd["crouch"] = r_crouch

    # 5. Height baseline
    r_height = 0.2 * max(0.0, z - 1.0)
    reward += r_height;  bd["height"] = r_height

    # 6. Posture (fall penalty)
    r_posture = -3.0 if body_fell else 0.0
    reward += r_posture;  bd["posture"] = r_posture

    # 7. Stay centered
    drift = np.sqrt(x ** 2 + y ** 2)
    r_drift = -0.1 * drift
    reward += r_drift;  bd["drift"] = r_drift

    # 8. Jerk penalty
    jerk = action - 2.0 * action_hist[-1] + action_hist[-2]
    r_jerk = -0.02 * np.sum(np.square(jerk))
    reward += r_jerk;  bd["jerk"] = r_jerk

    # 9. Control cost
    r_ctrl = -0.005 * np.sum(np.square(action))
    reward += r_ctrl;  bd["control"] = r_ctrl

    return reward, bd


# =============================================================================
# BACKFLIP REWARDS  (composable function)
# =============================================================================
def compute_backflip_rewards(z, airborne, quat, angular_vel_y,
                             cumulative_pitch, foot_max_z,
                             grounded, body_fell,
                             landed_after_flip, rotation_complete):
    """
    Backflip-specific rewards.  Returns (total, breakdown_dict).

    Components:
      ang_velocity  — reward backward pitch angular velocity while airborne
      foot_height   — reward feet being high (above torso → inverted body)
      rotation      — reward cumulative pitch progress toward 2π
      tuck          — reward compact body during mid-rotation
      landing       — big bonus for upright landing after ≥ 0.8 rotations
      completion    — one-time bonus when ≈ full rotation is achieved
      off_axis      — penalise roll / yaw (keep the flip in the sagittal plane)
    """
    bd = {}
    reward = 0.0

    # 1. Pitch angular velocity (backward = positive in MuJoCo Humanoid)
    r_ang = 0.0
    if airborne and z > 0.8:
        r_ang = 2.0 * np.clip(angular_vel_y, 0.0, 8.0)
    reward += r_ang;  bd["ang_velocity"] = r_ang

    # 2. Foot height — feet above torso means body is inverted
    r_foot = 0.0
    if airborne and z > 0.8:
        feet_above_torso = max(0.0, foot_max_z - z)
        r_foot = 5.0 * feet_above_torso
        r_foot += 1.0 * max(0.0, foot_max_z - STANDING_Z)
    reward += r_foot;  bd["foot_height"] = r_foot

    # 3. Rotation progress toward 2π
    r_rot = 0.0
    if airborne:
        frac = min(abs(cumulative_pitch) / (2.0 * np.pi), 1.0)
        r_rot = 3.0 * frac
        if frac > 0.75:
            r_rot += 5.0 * (frac - 0.75)
    reward += r_rot;  bd["rotation"] = r_rot

    # 4. Tuck bonus (compact body → faster spin)
    r_tuck = 0.0
    if airborne and z > 0.8 and abs(cumulative_pitch) > 0.5:
        tuck = max(0.0, STANDING_Z - z + 0.2)
        r_tuck = 0.5 * min(tuck, 0.4)
    reward += r_tuck;  bd["tuck"] = r_tuck

    # 5. Landing bonus — upright landing after ≥ 80 % rotation
    r_land = 0.0
    if landed_after_flip and not body_fell:
        rot_frac = abs(cumulative_pitch) / (2.0 * np.pi)
        if rot_frac > 0.8:
            _, pitch, _ = quat_to_euler(quat)
            uprightness = np.cos(pitch)
            r_land = 20.0 * max(0.0, uprightness) * min(rot_frac, 1.2)
    reward += r_land;  bd["landing"] = r_land

    # 6. Rotation completion (one-time)
    r_comp = 15.0 if rotation_complete else 0.0
    reward += r_comp;  bd["completion"] = r_comp

    # 7. Off-axis penalty (penalise roll & yaw during flight)
    r_off = 0.0
    if airborne:
        roll, _, yaw = quat_to_euler(quat)
        r_off = -0.5 * (abs(roll) + abs(yaw))
    reward += r_off;  bd["off_axis"] = r_off

    return reward, bd


# =============================================================================
# BackflipRewardWrapper
# =============================================================================
class BackflipRewardWrapper(gym.Wrapper):
    """
    Backflip = jump rewards (scaled) + backflip rewards (scaled).

    Approximate per-step reward gradient:
        standing still     →  ~0.08
        crouching          →  ~0.3
        jumping up         →  ~2.0   (jump rewards)
        rotating in air    →  ~6.0   (ang_vel + rotation progress)
        inverted (feet up) →  ~10.0  (foot height + rotation)
        landing after flip →  ~35.0  (landing + completion bonus)
    """

    def __init__(self, env, max_episode_steps=500,
                 jump_reward_scale=1.0, backflip_reward_scale=1.0):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.jump_reward_scale = jump_reward_scale
        self.backflip_reward_scale = backflip_reward_scale
        self._reset_tracking()

    # ── tracking state ───────────────────────────────────────────────────
    def _reset_tracking(self):
        self.step_count = 0
        self.max_z = -np.inf
        self.total_flight_steps = 0
        self.max_flight_z = -np.inf
        self.entered_flight = False

        # jump tracking
        self.was_crouching = False
        self.min_z_this_attempt = STANDING_Z
        self.was_grounded = True
        self.vz_at_liftoff = 0.0
        self.current_jump_steps = 0
        self.action_hist = [
            np.zeros(self.action_space.shape),
            np.zeros(self.action_space.shape),
        ]

        # backflip tracking
        self.cumulative_pitch = 0.0
        self.prev_pitch = 0.0
        self.max_foot_z = 0.0
        self.was_in_flight = False
        self.rotation_complete_given = False
        self.best_rotation = 0.0

        self.last_reward_breakdown = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_tracking()
        return obs, info

    # ── contact helpers (shared with jump_env) ───────────────────────────
    def _get_floor_contacts(self):
        d = self.unwrapped.data
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if FLOOR_GEOM_ID in pair:
                touching.add((pair - {FLOOR_GEOM_ID}).pop())
        return touching

    def _feet_on_ground(self, fc):
        return len(FOOT_GEOM_IDS & fc)

    def _non_foot_on_ground(self, fc):
        return len(fc - FOOT_GEOM_IDS) > 0

    def _get_foot_positions(self):
        d = self.unwrapped.data
        m = self.unwrapped.model
        try:
            rf_z = d.xpos[m.body(RIGHT_FOOT_BODY).id][2]
            lf_z = d.xpos[m.body(LEFT_FOOT_BODY).id][2]
        except Exception:
            rf_z = d.geom_xpos[RIGHT_FOOT_GEOM_ID][2]
            lf_z = d.geom_xpos[LEFT_FOOT_GEOM_ID][2]
        return rf_z, lf_z

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── gather state ──
        x, y, z = self.unwrapped.data.qpos[0:3]
        vx, vy, vz = self.unwrapped.data.qvel[0:3]
        quat = self.unwrapped.data.qpos[3:7]         # [w, x, y, z]
        angular_vel = self.unwrapped.data.qvel[3:6]   # [wx, wy, wz]
        angular_vel_y = angular_vel[1]

        floor_contacts = self._get_floor_contacts()
        n_feet = self._feet_on_ground(floor_contacts)
        body_fell = self._non_foot_on_ground(floor_contacts)
        airborne = (n_feet == 0) and (not body_fell)
        grounded = (n_feet > 0)

        self.max_z = max(self.max_z, z)

        # ── jump tracking (crouch / liftoff) ──
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
            # reset rotation tracking at each new liftoff
            self.cumulative_pitch = 0.0
            self.prev_pitch = quat_to_pitch(quat)
            self.rotation_complete_given = False
        self.was_grounded = grounded

        if airborne and z > 0.8:
            self.current_jump_steps += 1
            self.total_flight_steps += 1
            self.max_flight_z = max(self.max_flight_z, z)
            self.entered_flight = True

        # ── rotation tracking ──
        if airborne:
            dt = self.unwrapped.dt  # Gymnasium natively handles timestep * frame_skip here
            self.cumulative_pitch += angular_vel_y * dt
        self.prev_pitch = quat_to_pitch(quat)

        # ── foot height ──
        rf_z, lf_z = self._get_foot_positions()
        foot_max_z = max(rf_z, lf_z)
        self.max_foot_z = max(self.max_foot_z, foot_max_z)

        # ── landing detection ──
        landed_after_flip = (self.was_in_flight and grounded
                             and not body_fell)
        self.was_in_flight = airborne

        # ── rotation completion (one-time) ──
        rotation_complete = False
        if (abs(self.cumulative_pitch) > 1.8 * np.pi
                and not self.rotation_complete_given):
            self.rotation_complete_given = True
            rotation_complete = True
        self.best_rotation = max(self.best_rotation,
                                 abs(self.cumulative_pitch))

        # ══════════════════════════════════════════════════════════════════
        # REWARDS
        # ══════════════════════════════════════════════════════════════════
        r_jump, bd_jump = compute_jump_rewards(
            z=z, vz=vz, airborne=airborne, grounded=grounded,
            body_fell=body_fell, x=x, y=y,
            action=action, action_hist=self.action_hist,
            current_jump_steps=self.current_jump_steps,
            vz_at_liftoff=self.vz_at_liftoff,
            min_z_this_attempt=self.min_z_this_attempt,
        )

        r_flip, bd_flip = compute_backflip_rewards(
            z=z, airborne=airborne, quat=quat,
            angular_vel_y=angular_vel_y,
            cumulative_pitch=self.cumulative_pitch,
            foot_max_z=foot_max_z,
            grounded=grounded, body_fell=body_fell,
            landed_after_flip=landed_after_flip,
            rotation_complete=rotation_complete,
        )

        reward = (self.jump_reward_scale * r_jump
                  + self.backflip_reward_scale * r_flip)

        # merged breakdown
        bd = {}
        bd.update(bd_jump)
        bd.update({f"flip_{k}": v for k, v in bd_flip.items()})
        bd["jump_subtotal"] = r_jump
        bd["flip_subtotal"] = r_flip
        bd["total"] = reward
        self.last_reward_breakdown = bd

        # ── episode end ──
        drift = np.sqrt(x ** 2 + y ** 2)
        is_done = (terminated or truncated
                   or self.step_count >= self.max_episode_steps)
        if is_done:
            if self.entered_flight:
                info["jump_height"] = self.max_flight_z - STANDING_Z
                info["flight_steps"] = self.total_flight_steps
                info["best_rotation_deg"] = np.degrees(self.best_rotation)
                info["max_foot_z"] = self.max_foot_z
            if not terminated and not truncated:
                truncated = True

        # ── termination (more lenient than pure-jump) ──
        if z < 0.2:
            terminated = True
            reward -= 3.0
        if drift > 2.0:
            terminated = True
            reward -= 2.0

        # update action history
        self.action_hist[-2] = self.action_hist[-1].copy()
        self.action_hist[-1] = action.copy()

        # info for logging / visualisation
        info["z"] = z
        info["vz"] = vz
        info["airborne"] = airborne
        info["drift"] = drift
        info["cumulative_pitch"] = self.cumulative_pitch
        info["cumulative_pitch_deg"] = np.degrees(self.cumulative_pitch)
        info["foot_max_z"] = foot_max_z
        info["angular_vel_y"] = angular_vel_y
        info["reward_breakdown"] = bd

        return obs, reward, terminated, truncated, info
