import gymnasium as gym
import numpy as np


class LimitedFuelAndImpactWrapper(gym.Wrapper):
    """
    Standardized wrapper enforcing fuel limits, landing safety bonuses,
    and progressive penalties for excessive continuous thruster usage.
    """

    def __init__(self, env, max_fuel=200.0, max_safe_landing_speed=0.7):
        super().__init__(env)
        self.max_fuel = max_fuel
        self.fuel = max_fuel
        self.max_safe_landing_speed = max_safe_landing_speed
        self.prev_vertical_velocity = 0.0
        self.thruster_streak = 0

    def reset(self, **kwargs):
        self.fuel = self.max_fuel
        self.thruster_streak = 0
        state, info = self.env.reset(**kwargs)
        self.prev_vertical_velocity = state[3]
        return state, info

    def step(self, action):
        # Disable engines when fuel is completely depleted
        if self.fuel <= 0:
            action = 0

        # Deduct fuel per active engine action
        if action == 2:          # Main engine
            self.fuel -= 0.8
        elif action in (1, 3):   # Side engines
            self.fuel -= 0.2

        # Track continuous thruster activation duration
        if action in (1, 2, 3):
            self.thruster_streak += 1
        else:
            self.thruster_streak = 0

        state, reward, terminated, truncated, info = self.env.step(action)

        # Apply progressive penalty for continuous thruster overuse
        

        pre_impact_speed = abs(self.prev_vertical_velocity)
        current_vertical_velocity = state[3]

        if terminated or truncated:
            both_legs_touching = bool(state[6] and state[7])

            if (
                both_legs_touching
                and pre_impact_speed <= self.max_safe_landing_speed
            ):
                # Base reward for safe landing
                reward += 150.0

                # Fuel bonus: reward remaining fuel in tank upon successful landing
                fuel_bonus = self.fuel * 2.0
                reward += fuel_bonus

                info["impact_status"] = (
                    f"PERFECT LANDING! Speed: {pre_impact_speed:.2f} | "
                    f"Fuel Bonus: +{fuel_bonus:.1f}"
                )
            else:
                reward -= 50.0
                info["impact_status"] = f"CRASH! Speed: {pre_impact_speed:.2f}"
        else:
            info["impact_status"] = (
                f"Flying | Speed: {abs(current_vertical_velocity):.2f}"
            )

        self.prev_vertical_velocity = current_vertical_velocity
        info["fuel_left"] = max(0.0, self.fuel)
        info["vertical_velocity"] = current_vertical_velocity

        return state, reward, terminated, truncated, info