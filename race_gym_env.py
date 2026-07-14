"""
Gymnasium Environment for Race Simulation
Implements the Gymnasium API for DRL training on race simulation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional
from domain_model import Race, Car, Track, TireType
from race_config import setup_real_race
from simulation_logger import SimulationLogger


class RaceGymEnv(gym.Env):
    """
    Gymnasium Environment for Race Simulation
    
    Action Space: Discrete(4) - Agent pit decision (no pit / pit+soft / pit+medium / pit+hard)
    
    Observation Space: Dict with:
        - lap_time: Current lap time
        - race_progress: Current lap / total laps (0.0 to 1.0)
        - lap: Current lap number (normalized)
        - tire_wear: Current tire wear (0.0 to 1.0)
        - fuel_mass: Current fuel mass
        - competitors_info: Information about other cars (relative time differences)
    """
    
    def __init__(
        self,
        gp_data: dict,
        render_mode: Optional[str] = None,
        logger: Optional[SimulationLogger] = None,
        agent_car_id: int = 1,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.agent_car_id = agent_car_id
        self.race = setup_real_race(gp_data, agent_car_id=agent_car_id)
        self.num_competitors = len(self.race.cars) - 1
        self.num_cars = len(self.race.cars)

        # Focus on configured car as the agent
        self.agent_car = next(c for c in self.race.cars if c.car_id == self.agent_car_id)
        
        # Calculate initial fuel mass for normalization
        self.initial_fuel_mass = self.race.fuel_model.initial_fuel_mass(self.race.track)

        # Define action and observation spaces
        # 0: no pit, 1: pit + soft, 2: pit + medium, 3: pit + hard
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Dict({
            'lap_time': spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
            'race_progress': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'tire_wear': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'fuel_mass': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'competitors_info': spaces.Box(low=-1.0, high=1.0, shape=(self.num_competitors,), dtype=np.float32),
            'tire_type': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        
        # Track episode statistics
        self.previous_lap = 1

        # Logger
        self.logger = logger
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        
        race_progress = self.agent_car.lap / self.race.track.laps

        # tire_wear: laps_on_tire normalizado por el total de vueltas de la carrera
        tire_wear = min(1.0, self.agent_car.laps_on_tire / self.race.track.laps)

        # Get competitor information (relative time differences, sorted by real accumulated time)
        sorted_cars = sorted(self.race.cars, key=lambda c: c.total_time)
        try:
            agent_pos = [c.car_id for c in sorted_cars].index(self.agent_car_id)
        except ValueError:
            agent_pos = len(self.race.cars) - 1

        cars_ahead = sorted_cars[:agent_pos]
        cars_behind = sorted_cars[agent_pos + 1:]

        gaps_ahead = [(car.total_time - self.agent_car.total_time) / 300.0 for car in cars_ahead]
        gaps_behind = [(car.total_time - self.agent_car.total_time) / 300.0 for car in cars_behind]

        # Pad or truncate to fixed size (e.g., 10 ahead, 9 behind)
        padded_ahead = ([-1.0] * (10 - len(gaps_ahead)) + gaps_ahead) if len(gaps_ahead) < 10 else gaps_ahead[-10:]
        padded_behind = (gaps_behind + [1.0] * (9 - len(gaps_behind))) if len(gaps_behind) < 9 else gaps_behind[:9]

        competitors_info = np.array(padded_ahead + padded_behind, dtype=np.float32)
        competitors_info = np.clip(competitors_info, -1.0, 1.0)

        # One-hot encoding of current tire type (SOFT, MEDIUM, HARD)
        tire_type_vec = np.zeros(3, dtype=np.float32)
        if self.agent_car.current_tire_type == TireType.SOFT:
            tire_type_vec[0] = 1.0
        elif self.agent_car.current_tire_type == TireType.MEDIUM:
            tire_type_vec[1] = 1.0
        elif self.agent_car.current_tire_type == TireType.HARD:
            tire_type_vec[2] = 1.0

        # Normalizar fuel_mass con respecto al combustible inicial
        fuel_ratio = self.agent_car.fuel_mass / self.initial_fuel_mass if self.initial_fuel_mass > 0 else 0.0

        return {
            'lap_time': np.array([self.agent_car.current_lap_time], dtype=np.float32),
            'race_progress': np.array([race_progress], dtype=np.float32),
            'tire_wear': np.array([tire_wear], dtype=np.float32),
            'fuel_mass': np.array([fuel_ratio], dtype=np.float32),
            'competitors_info': competitors_info,
            'tire_type': tire_type_vec
        }
    
    def _calculate_reward(self, action: int = 0) -> float:
        """Calculate reward based on current state"""
        reward = 0.0
        
        # Lap time reward (faster lap times get higher rewards)
        if self.agent_car.current_lap_time > 0:
            # Normalize lap time (assuming 60 seconds as baseline)
            time_reward = max(0.1, 1.0 - (self.agent_car.current_lap_time - 60) / 240)
            reward += time_reward
        
        # Rank reward (being ahead of competitors, based on real leaderboard by accumulated time)
        sorted_cars = sorted(self.race.cars, key=lambda c: c.total_time)
        try:
            agent_rank = [car.car_id for car in sorted_cars].index(self.agent_car_id)
        except ValueError:
            agent_rank = len(self.race.cars) - 1
            
        num_ahead = agent_rank
        num_behind = len(self.race.cars) - 1 - agent_rank
        reward += num_behind * 0.05 - num_ahead * 0.03
        
        # Lap completion reward
        if self.agent_car.lap > self.previous_lap:
            reward += 1.0
        self.previous_lap = self.agent_car.lap
        
        # Finish reward
        if self.agent_car.is_finished:
            if self.race.winner and self.race.winner.car_id == self.agent_car_id:
                reward += 10.0  # Win bonus
            else:
                reward += 5.0   # Finish bonus
            
            # F1 rule check: must use at least two different dry tire compounds
            unique_compounds = set(self.agent_car.tire_history)
            if len(unique_compounds) < 2:
                reward -= 50.0  # Severe penalty for failing to use at least two different compounds
                
        # Pit stop penalty to represent the 23-second physical cost of pitting
        if action in (1, 2, 3):
            reward -= 0.5  # Moderate penalty to prevent excessive pitting without hiding long-term gains
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode is terminated"""
        return self.agent_car.is_finished or self.race.race_finished
    
    def _is_truncated(self) -> bool:
        """Check if episode is truncated"""
        return False
    
    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Convert action to standard Python integer to avoid unhashable numpy array issues
        if hasattr(action, "item"):
            action_int = int(action.item())
        else:
            action_int = int(action)

        # Step the race (cars move automatically)
        race_state = self.race.step_with_action(agent_action=action_int)

        # Enriquecer race_state con el compuesto actual de cada coche
        # (necesario para el logger; Race no lo incluye por defecto)
        race_state["tire_type"] = {
            car.car_id: car.current_tire_type.value
            for car in self.race.cars
        }

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action_int)

        # Check termination and truncation
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Logger
        if self.logger is not None:
            self.logger.log_lap(
                race_state=race_state,
                agent_car_id=self.agent_car_id,
                agent_action=action_int,
                track_laps=self.race.track.laps,
            )
            if terminated or truncated:
                self.logger.close_episode(final_state=race_state)

        # Info dictionary
        info = {
            'race_state': race_state,
            'episode_step': self.race.time_step,
            'agent_lap': self.agent_car.lap,
            'winner': race_state.get('winner')
        }

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)

        # Reset race
        self.race.reset()

        # Reset episode tracking
        self.previous_lap = 1

        # Notificar al logger del nuevo episodio
        if self.logger is not None:
            self.logger.on_reset(track_laps=self.race.track.laps)

        # Get initial observation
        observation = self._get_observation()

        info = {'race_state': self.race.get_race_state()}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _render_frame(self):
        """Render the current frame (simple text-based)"""
        if self.render_mode != "human":
            return
        
        # Simple text rendering
        print(f"\n=== Race Step {self.race.time_step} (Lap {self.agent_car.lap}) ===")
        print(f"Agent Car ({self.agent_car.driver_name}):")
        print(f"  Lap Time: {self.agent_car.current_lap_time:.2f} seconds")
        print(f"  Total Time: {self.agent_car.total_time:.2f} seconds")
        print(f"  Lap: {self.agent_car.lap}/{self.race.track.laps}")
        tire_wear = min(1.0, self.agent_car.laps_on_tire / self.race.track.laps)
        print(f"  Tire Wear: {tire_wear:.2f}")
        print(f"  Fuel Mass: {getattr(self.agent_car, 'fuel_mass', 0.0):.2f}")
        print(f"  Finished: {self.agent_car.is_finished}")

        print("\nCompetitors:")
        for car in self.race.cars:
            if car.car_id == self.agent_car_id:
                continue
            car_wear = min(1.0, car.laps_on_tire / self.race.track.laps)
            print(
                f"  Car {car.car_id}: Total={car.total_time:.2f}s "
                f"(Lap {car.lap}, LapTime={car.current_lap_time:.2f}s, Wear={car_wear:.2f})"
            )
        
        if self.race.winner:
            print(f"\nWinner: Car {self.race.winner.car_id}")
    
    def close(self):
        """Close the environment"""
        if self.logger is not None:
            self.logger.close()


# Registration function for Gymnasium registry
def register_race_env():
    """Register the Race environment with Gymnasium"""
    gym.register(
        id='RaceSim-v0',
        entry_point=RaceGymEnv,
        max_episode_steps=1000,
        reward_threshold=10.0,
    )


# Example usage and testing
if __name__ == "__main__":
    # Register the environment
    register_race_env()
    
    # Create environment
    env = gym.make('RaceSim-v0', render_mode="human")
    
    # Test the environment
    print("Testing Race Environment...")
    
    obs, info = env.reset()
    print("Initial observation shape:", {k: v.shape for k, v in obs.items()})
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    # Run a few random steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Action={action}, Reward={reward:.2f}")
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    env.close()
