import gymnasium as gym
import numpy as np


class LimitedFuelAndImpactWrapper(gym.Wrapper):
    """
    Standardized wrapper enforcing extended fuel limits (500.0), 
    scaled fuel efficiency rewards, strict center pad alignment (|X| <= 0.08),
    extreme out-of-bounds penalties, and dual-leg + zero-thruster stability.
    """

    def __init__(self, env, max_fuel=200.0, max_safe_landing_speed=0.7, required_hold_steps=10):
        super().__init__(env)
        self.max_fuel = max_fuel
        self.fuel = max_fuel
        self.max_safe_landing_speed = max_safe_landing_speed
        self.required_hold_steps = required_hold_steps
        self.prev_vertical_velocity = 0.0
        self.thruster_streak = 0
        self.landing_hold_counter = 0

        # Expand observation space bounds to 9 elements (8 original + 1 fuel ratio)
        low = np.append(self.env.observation_space.low, 0.0)
        high = np.append(self.env.observation_space.high, 1.0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def _get_extended_state(self, state):
        """Appends normalized fuel level (0.0 to 1.0) to the state vector."""
        fuel_ratio = max(0.0, self.fuel / self.max_fuel)
        return np.append(state, fuel_ratio).astype(np.float32)

    def reset(self, **kwargs):
        self.fuel = self.max_fuel
        self.thruster_streak = 0
        self.landing_hold_counter = 0
        state, info = self.env.reset(**kwargs)
        self.prev_vertical_velocity = state[3]
        return self._get_extended_state(state), info

    def step(self, action):
        # Disable engines when fuel is completely depleted
        if self.fuel <= 0:
            action = 0

        # Deduct fuel per active engine action
        if action == 2:  # Main engine
            self.fuel -= 0.8
        elif action in (1, 3):  # Side engines
            self.fuel -= 0.2

        # Track continuous thruster activation duration
        if action in (1, 2, 3):
            self.thruster_streak += 1
        else:
            self.thruster_streak = 0

        state, reward, terminated, truncated, info = self.env.step(action)

        # Dual leg contact requirement & engine cutoff verification
        both_legs_contact = bool(state[6] and state[7])
        any_leg_contact = bool(state[6] or state[7])
        no_thrusters_active = (action == 0)
        
        horizontal_pos = state[0]
        horizontal_dist = abs(horizontal_pos)

        # 1. Narrow Centering Reward (|X| <= 0.08 strict center pad)
        if horizontal_dist > 0.08:
            reward -= horizontal_dist * 3.0  # Stronger penalty for straying off center
        else:
            reward += (0.08 - horizontal_dist) * 2.5  # High bonus for precise alignment

        # 2. Strict Side/Main Thruster Penalty on Ground
        if any_leg_contact:
            reward += 1.0
            if action in (1, 3):  # Firing side engines on ground
                reward -= 5.0
            elif action == 2:     # Firing main engine on ground
                reward -= 2.0

        # 3. Anti-Hovering & Final Touchdown Incentives
        if horizontal_dist <= 0.08:
            if state[1] < 0.4 and state[3] < 0:
                reward += abs(state[3]) * 1.0

            if state[1] < 0.20 and not any_leg_contact:
                reward -= state[1] * 2.0

        # Apply progressive penalty for continuous thruster overuse
        if self.thruster_streak > 0:
            excessive_thruster_penalty = 0.05 * (
                1.1 ** min(self.thruster_streak, 30)
            )
            reward -= excessive_thruster_penalty

        pre_impact_speed = abs(self.prev_vertical_velocity)
        current_vertical_velocity = state[3]

        near_ground = state[1] <= 0.15
        true_ground_touchdown = both_legs_contact and near_ground and no_thrusters_active
        is_centered = horizontal_dist <= 0.08

        # Evaluate stable positioning strictly inside narrow target zone with BOTH legs and NO thrusters
        if true_ground_touchdown and is_centered and pre_impact_speed <= self.max_safe_landing_speed:
            self.landing_hold_counter += 1
        else:
            self.landing_hold_counter = 0

        # TERMINATE ONLY IF HELD STABLE WITH BOTH LEGS AND ZERO ENGINES FOR REQUIRED STEPS
        if self.landing_hold_counter >= self.required_hold_steps:
            terminated = True
            reward += 200.0

            # High-scaled fuel bonus: reward remaining fuel out of 500.0 max capacity
            fuel_bonus = self.fuel * 4.0
            reward += fuel_bonus

            info["impact_status"] = (
                f"PERFECT LANDING! Speed: {pre_impact_speed:.2f} | "
                f"Fuel Bonus: +{fuel_bonus:.1f}"
            )
        elif truncated:
            # EXTREME PENALTY: Flying off-screen / out of bounds
            reward -= 300.0
            info["impact_status"] = f"CRASH (OUT OF BOUNDS)! Speed: {pre_impact_speed:.2f}"
        elif terminated:
            # Penalty for crashing or landing outside the safe zone
            reward -= 50.0
            if not is_centered and (both_legs_contact and near_ground):
                info["impact_status"] = f"CRASH (OFF-TARGET)! Speed: {pre_impact_speed:.2f}"
            else:
                info["impact_status"] = f"CRASH! Speed: {pre_impact_speed:.2f}"
        else:
            info["impact_status"] = (
                f"Flying | Speed: {abs(current_vertical_velocity):.2f}"
            )

        self.prev_vertical_velocity = current_vertical_velocity
        info["fuel_left"] = max(0.0, self.fuel)
        info["vertical_velocity"] = current_vertical_velocity

        extended_state = self._get_extended_state(state)
        return extended_state, reward, terminated, truncated, info