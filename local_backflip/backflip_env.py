import gymnasium as gym
import numpy as np

FLOOR_GEOM_ID      = 0
RIGHT_FOOT_GEOM_ID = 8
LEFT_FOOT_GEOM_ID  = 11
FOOT_GEOM_IDS      = {RIGHT_FOOT_GEOM_ID, LEFT_FOOT_GEOM_ID}

STANDING_Z = 1.4
FULL_BACKFLIP = 2.0 * np.pi   


def quat_to_pitch(qw, qx, qy, qz):
    """Extract pitch angle (rotation about y-axis) from quaternion.
    Returns value in [-pi, pi]. Negative = pitched backward."""
    sin_p = 2.0 * (qw * qy - qz * qx)
    sin_p = np.clip(sin_p, -1.0, 1.0)
    return np.arcsin(sin_p)


class BackflipRewardWrapper(gym.Wrapper):
    """
    Phase-aware backflip reward with contact-based flight detection.

    Rough per-step reward scale:
        standing still         →  ~0.05  (just height baseline)
        crouching to load      →  ~0.3   (crouch bonus)
        launching (vz > 0)     →  ~0.5   (velocity reward)
        airborne + rotating    →  ~3-8   (flight + angular vel + progress)
        landing after 360°     →  +50    (one-time completion bonus)
    """

    def __init__(self, env, max_episode_steps=500):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.dt = self.unwrapped.dt  
        self._reset_tracking()

    def _reset_tracking(self):
        self.step_count = 0

        self.max_z = -np.inf
        self.max_flight_z = -np.inf

        self.total_flight_steps = 0
        self.entered_flight = False
        self.was_grounded = True
        self.cumulative_pitch = 0.0      
        self.max_abs_pitch = 0.0         
        self.prev_pitch = 0.0             
        self.backflip_completed = False  

        self.was_crouching = False

        self.last_reward_breakdown = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_tracking()
        return obs, info

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

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        qpos = self.unwrapped.data.qpos
        qvel = self.unwrapped.data.qvel

        x, y, z = qpos[0:3]
        vx, vy, vz = qvel[0:3]
        wx, wy, wz = qvel[3:6]         

        qw, qx, qy, qz = qpos[3:7]    
        pitch = quat_to_pitch(qw, qx, qy, qz)

        floor_contacts = self._get_floor_contacts()
        n_feet = self._feet_on_ground(floor_contacts)
        body_fell = self._non_foot_on_ground(floor_contacts)
        grounded = (n_feet > 0)
        airborne = (n_feet == 0) and (not body_fell)

        self.max_z = max(self.max_z, z)

        self.cumulative_pitch += wy * self.dt
        backward_rotation = abs(self.cumulative_pitch)
        self.max_abs_pitch = max(self.max_abs_pitch, backward_rotation)

        if backward_rotation >= FULL_BACKFLIP * 0.9 and not self.backflip_completed:
            self.backflip_completed = True

        if self.was_grounded and not grounded and not body_fell:
            pass  
        self.was_grounded = grounded

        reward = 0.0
        bd = {}

        r_vel = 0.5 * np.clip(vz, 0.0, 5.0)
        reward += r_vel
        bd["z_velocity"] = r_vel

        r_height = 0.0
        if airborne and z > 0.8:
            height_above = max(0.0, z - STANDING_Z)
            r_height += 1.0                       
            r_height += 3.0 * height_above        
            r_height += 5.0 * height_above ** 2   

            self.total_flight_steps += 1
            self.max_flight_z = max(self.max_flight_z, z)
            self.entered_flight = True
        reward += r_height
        bd["z_height"] = r_height

        r_angvel = 0.0
        if airborne and z > 0.8:
            
            backward_speed = max(-wy, 0.0)        
            r_angvel = 2.0 * np.clip(backward_speed, 0.0, 8.0)
        reward += r_angvel
        bd["angular_velocity"] = r_angvel

        r_rotation = 0.0
        if airborne and z > 0.8:
            progress = min(backward_rotation / FULL_BACKFLIP, 1.0)  # 0 → 1
            r_rotation = 5.0 * progress
        reward += r_rotation
        bd["rotation_progress"] = r_rotation

        r_landing = 0.0
        if self.backflip_completed and grounded and not body_fell and z > 1.0:
            r_landing = 50.0
            self.backflip_completed = False  
        reward += r_landing
        bd["landing_bonus"] = r_landing

        r_crouch = 0.0
        if grounded and z < STANDING_Z - 0.05 and z > 0.8:
            crouch_depth = STANDING_Z - z
            r_crouch = 0.3 * min(crouch_depth, 0.3)
        reward += r_crouch
        bd["crouch"] = r_crouch

        r_posture = -3.0 if body_fell else 0.0
        reward += r_posture
        bd["posture"] = r_posture

        drift = np.sqrt(x**2 + y**2)
        r_drift = -0.1 * drift
        reward += r_drift
        bd["drift"] = r_drift

        r_ctrl = -0.005 * np.sum(np.square(action))
        reward += r_ctrl
        bd["control"] = r_ctrl

        bd["total"] = reward
        self.last_reward_breakdown = bd

        is_done = terminated or truncated or (self.step_count >= self.max_episode_steps)
        if is_done:
            info["max_rotation_deg"] = np.degrees(self.max_abs_pitch)
            info["backflip_completed"] = self.backflip_completed
            if self.entered_flight:
                info["jump_height"] = self.max_flight_z - STANDING_Z
                info["flight_steps"] = self.total_flight_steps
            if not terminated and not truncated:
                truncated = True

        if z < 0.3:
            terminated = True
            reward -= 3.0
        if drift > 2.0:
            terminated = True
            reward -= 2.0

        info["z"] = z
        info["vz"] = vz
        info["airborne"] = airborne
        info["drift"] = drift
        info["pitch_deg"] = np.degrees(pitch)
        info["cumulative_pitch_deg"] = np.degrees(self.cumulative_pitch)
        info["backward_rotation_deg"] = np.degrees(backward_rotation)
        info["angular_vel_y"] = wy
        info["reward_breakdown"] = bd

        return obs, reward, terminated, truncated, info