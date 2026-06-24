"""
Training script for DRL agent using Stable-Baselines3
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from race_gym_env import RaceGymEnv, register_race_env
from race_config import load_gp_from_json
import matplotlib.pyplot as plt
from evaluate_agent import evaluate_model


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress"""
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_positions = []
        self.episode_timesteps = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_reward += self.locals.get('rewards')[0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals.get('dones')[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_timesteps.append(self.num_timesteps)
            
            # Find agent's final position in the race
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0:
                race_state = infos[0].get('race_state', {})
                leaderboard = race_state.get('leaderboard_by_time', [])
                try:
                    rank = [car_id for car_id, _ in leaderboard].index(0) + 1
                except ValueError:
                    rank = 20  # fallback
            else:
                rank = 20
            self.episode_positions.append(rank)
            
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Reward={self.current_episode_reward:.2f}, "
                      f"Position={rank}, ")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True
        
    def _on_rollout_end(self) -> None:
        """Called at the end of every rollout collection. Record mean position to logger."""
        if len(self.episode_positions) > 0:
            # Mean position of the last 100 episodes (same window size as SB3 reward tracking)
            mean_pos = np.mean(self.episode_positions[-100:])
            self.logger.record("rollout/ep_position_mean", mean_pos)
    
    def get_training_stats(self):
        """Return training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_positions': self.episode_positions,
            'episode_timesteps': self.episode_timesteps
        }


def moving_average(data, window_size=50):
    """Calculate moving average with a sliding window, keeping the output size matching the input size."""
    data = np.asarray(data)
    if len(data) == 0:
        return data
    result = np.zeros_like(data, dtype=float)
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        result[i] = np.mean(data[start:i + 1])
    return result


def plot_training_progress(callback):
    """Plot training progress using matplotlib with clean and unified styling."""
    stats = callback.get_training_stats()
    rewards = stats['episode_rewards']
    positions = stats['episode_positions']
    timesteps = stats.get('episode_timesteps')
    
    if not timesteps or len(timesteps) == 0:
        timesteps = list(range(len(rewards)))
    
    if len(rewards) == 0:
        print("No training statistics available to plot.")
        return
        
    # Configure matplotlib aesthetics
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    
    # Calculate moving averages (adapt window size based on amount of data)
    window_size = min(50, max(5, len(rewards) // 10))
    smoothed_rewards = moving_average(rewards, window_size=window_size)
    smoothed_positions = moving_average(positions, window_size=window_size)
    
    # ------------------
    # Plot 1: Cumulative Reward Curve
    # ------------------
    # Raw episode rewards in a lighter color
    ax1.plot(timesteps, rewards, color='#a1c9f4', alpha=0.4, label='Raw Episode Reward')
    # Smoothed moving average in a thicker, darker color
    ax1.plot(timesteps, smoothed_rewards, color='#1f77b4', linewidth=2.5, 
             label=f'Moving Average (window={window_size})')
    
    ax1.set_title('Cumulative Reward Curve', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('Total Reward per Episode', fontsize=12)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # ------------------
    # Plot 2: Evolution of the Average Final Position
    # ------------------
    # Raw final positions
    ax2.plot(timesteps, positions, color='#ffbeb2', alpha=0.4, label='Raw Final Position')
    # Smoothed final position
    ax2.plot(timesteps, smoothed_positions, color='#d62728', linewidth=2.5, 
             label=f'Moving Average (window={window_size})')
    
    ax2.set_title('Evolution of the Average Final Position', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Final Position (1st at Top)', fontsize=12)
    
    # Y-axis configuration: 1 to 20, with 1 at the top
    ax2.set_ylim(20.5, 0.5)  # Inverted Y-axis so 1st place is at the top
    ax2.set_yticks(range(1, 21, 2))
    
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300)
    plt.show()


def main():
    """Main training function"""
    print("Starting DRL Training for Race Simulation...")
    
    # Register environment
    register_race_env()
    
    # Load real GP data
    gp_data = load_gp_from_json("data/spanish_gp_2024.json")
    print("Loaded real GP data: Spanish GP 2024")

    # Create environment
    env = RaceGymEnv(gp_data=gp_data)
    
    # Check environment
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return
    
    # Create callback for monitoring
    callback = TrainingCallback(verbose=1)
    
    # Choose algorithm and train
    print("\nTraining PPO agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=0.0002,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./race_tensorboard/"
    )
    
    # Train the model
    print("Training started...")
    model.learn(total_timesteps=500000, callback=callback)
    print("Training completed!")
    
    # Save the model
    model.save("race_agent_ppo")
    print("Model saved as 'race_agent_ppo.zip'")
    
    # Plot training progress
    plot_training_progress(callback)
    
    # Evaluate the trained model
    print("\nEvaluating trained model...")
    eval_rewards, eval_lengths, eval_positions = evaluate_model(model, env, n_episodes=20, deterministic=False, gp_data=gp_data)
    

    
    env.close()


def test_environment():
    """Test the environment without training"""
    print("Testing Race Environment...")
    
    # Load real GP data
    gp_data = load_gp_from_json("data/spanish_gp_2024.json")
    print("Loaded real GP data for testing.")
        
    env = RaceGymEnv(render_mode="human", gp_data=gp_data)
    
    # Test automatic movement
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        obs, info = env.reset()
        
        for step in range(env.race.track.laps):
            action = 0  # Dummy action, cars move automatically
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step == 0:
                print("\nSample Observation at Step 0:")
                for k, v in obs.items():
                    if k == 'competitors_info':
                        # Formatear la lista de rivales para que no ocupe demasiado
                        print(f"  {k}: {['%.4f' % val for val in v]}")
                    else:
                        print(f"  {k}: {v}")
            
            if terminated or truncated:
                print(f"Episode finished in {step + 1} steps!")
                break
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_environment()
    else:
        main()
