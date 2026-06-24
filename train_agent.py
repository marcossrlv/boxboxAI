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
            'episode_positions': self.episode_positions
        }


def plot_training_progress(callback):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot episode rewards
    ax1.plot(callback.episode_rewards)
    ax1.set_title('Episode Rewards During Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot final positions
    ax2.plot(callback.episode_positions)
    ax2.set_title('Final Position During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Position (Lower is Better)')
    ax2.invert_yaxis()  # 1st place at the top, 20th at the bottom
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
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
    eval_rewards, eval_lengths, eval_positions = evaluate_model(model, env, n_episodes=20, deterministic=False)
    

    
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
