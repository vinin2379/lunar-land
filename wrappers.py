import gymnasium as gym
import numpy as np

class LimitedFuelAndImpactWrapper(gym.Wrapper):
    """
    Custom Wrapper that limits fuel, penalizes hovering/climbing away from target,
    and rewards progress towards the helipad landing coordinates (0, 0).
    """
    def __init__(self, env, max_fuel=100.0, max_safe_landing_speed=0.7):
        super().__init__(env)
        self.max_fuel = max_fuel
        self.fuel = max_fuel
        self.max_safe_landing_speed = max_safe_landing_speed
        self.prev_vertical_velocity = 0.0
        self.prev_distance = None

    def reset(self, **kwargs):
        self.fuel = self.max_fuel
        self.prev_vertical_velocity = 0.0
        state, info = self.env.reset(**kwargs)
        self.prev_vertical_velocity = state[3]
        
        # Calculate initial distance to target center (x=0, y=0)
        self.prev_distance = np.sqrt(state[0]**2 + state[1]**2)
        return state, info

    def step(self, action):
        # Disable engine if fuel runs out
        if self.fuel <= 0:
            action = 0

        # Deduct fuel
        if action == 2:          # Main engine
            self.fuel -= 0.8
        elif action in (1, 3):   # Side engines
            self.fuel -= 0.2

        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Current distance to the landing pad center (0,0)
        current_distance = np.sqrt(state[0]**2 + state[1]**2)
        
        # Reward progress towards the center target, penalize moving away/boosting up
        distance_delta = self.prev_distance - current_distance
        reward += distance_delta * 10.0  # Positive if getting closer, negative if flying away
        self.prev_distance = current_distance

        # Penalize flying too high (prevent boosting to upper screen boundary)
        if state[1] > 1.0:
            reward -= 0.5

        current_vertical_velocity = state[3]

        if terminated or truncated:
            impact_speed = abs(self.prev_vertical_velocity)
            legs_touching = bool(state[6] or state[7])

            # Check if landing occurs close to the target center
            x_position = abs(state[0])
            centered_landing = x_position < 0.2

            if legs_touching and impact_speed <= self.max_safe_landing_speed and centered_landing:
                reward += 200.0  # Big bonus for soft & centered landing
                info["impact_status"] = f"PERFECT LANDING! Speed: {impact_speed:.2f}"
            elif legs_touching and impact_speed <= self.max_safe_landing_speed:
                reward += 100.0  # Bonus for soft landing outside center
                info["impact_status"] = f"SOFT LANDING! Speed: {impact_speed:.2f}"
            elif impact_speed > self.max_safe_landing_speed:
                excess_speed = impact_speed - self.max_safe_landing_speed
                impact_penalty = min(excess_speed * 15.0, 30.0)
                reward -= impact_penalty
                info["impact_status"] = f"CRASH! Speed: {impact_speed:.2f}"
            else:
                info["impact_status"] = f"TOUCHDOWN! Speed: {impact_speed:.2f}"
        else:
            info["impact_status"] = f"Flying | Speed: {abs(current_vertical_velocity):.2f}"

        self.prev_vertical_velocity = current_vertical_velocity
        info["fuel_left"] = max(0.0, self.fuel)
        info["vertical_velocity"] = current_vertical_velocity
        
        return state, reward, terminated, truncated, info