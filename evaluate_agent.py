"""
Evaluation script for trained DRL agents in Race Simulation
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from race_gym_env import RaceGymEnv, register_race_env
from race_config import load_gp_from_json


def evaluate_model(model, env, n_episodes, deterministic=True):
    """
    Evaluate a loaded model on a given environment.
    
    Args:
        model: The trained Stable-Baselines3 model.
        env: The Gymnasium environment.
        n_episodes: Number of episodes to run the evaluation.
        deterministic: Whether to use deterministic actions.
        
    Returns:
        tuple: (episode_rewards, episode_lengths, episode_positions)
    """
    episode_rewards = []
    episode_lengths = []
    episode_positions = []
    wins = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Track pit stops and compound changes
        starting_compound = env.agent_car.current_tire_type.value.upper()
        pit_stops = []
        
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Check if action is a pit stop (1, 2, or 3) before stepping
            action_val = int(action.item()) if hasattr(action, "item") else int(action)
            if action_val in (1, 2, 3):
                compounds = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}
                pit_stops.append((env.agent_car.lap, compounds[action_val]))
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                # Find agent's final position
                final_state = info.get('race_state', {})
                leaderboard = final_state.get('leaderboard_by_time', [])
                try:
                    rank = [car_id for car_id, _ in leaderboard].index(env.agent_car_id) + 1
                except ValueError:
                    rank = env.num_cars  # fallback
                episode_positions.append(rank)
                
                if info.get('winner') == 0:  # Agent won
                    wins += 1
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Format the strategy string
        strategy_str = starting_compound
        for lap, comp in pit_stops:
            strategy_str += f" -> [Lap {lap}: {comp}]"
            
        print(f"Eval Episode {episode + 1}: Position={episode_positions[-1]}, Reward={total_reward:.2f}, Steps={steps}, Strategy: {strategy_str}")

        # Print final leaderboard for this episode
        driver_names = {car.car_id: car.driver_name for car in env.race.cars}
        print(f"\n--- CLASIFICACIÓN FINAL (EPISODIO {episode + 1}) ---")
        print(f"{'Pos':>3} | {'Piloto':<25} | {'Car ID':>6} | {'Tiempo':>12}")
        print("-" * 55)
        
        winner_time = None
        for pos, (car_id, total_time) in enumerate(leaderboard):
            driver_name = driver_names.get(car_id, f"Driver {car_id}")
            # Highlight agent car
            if car_id == env.agent_car_id:
                driver_name = f"* {driver_name}"
            else:
                driver_name = f"  {driver_name}"
                
            if pos == 0:
                winner_time = total_time
                time_str = f"{total_time:.3f}s"
            else:
                gap = total_time - winner_time
                time_str = f"+{gap:.3f}s"
            print(f"{pos + 1:>3} | {driver_name:<25} | {car_id:>6} | {time_str:>12}")
        print("-" * 55 + "\n")
    
    print(f"\nEvaluation Results (Deterministic={deterministic}):")
    print(f"Average Final Position: {np.mean(episode_positions):.1f} ± {np.std(episode_positions):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Win Rate: {wins}/{n_episodes} ({wins/n_episodes*100:.1f}%)")
    
    return episode_rewards, episode_lengths, episode_positions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DRL race agent.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="race_agent_ppo", 
        help="Path to the saved model zip file (without .zip extension, default: race_agent_ppo)"
    )
    parser.add_argument(
        "--gp", 
        type=str, 
        default="data/spanish_gp_2024.json", 
        help="JSON file containing the Grand Prix configuration (default: data/spanish_gp_2024.json)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=20, 
        help="Number of episodes to evaluate (default: 20)"
    )
    parser.add_argument(
        "--stochastic", 
        action="store_true", 
        help="Use stochastic actions instead of deterministic actions"
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Render the race steps (text-based render mode 'human')"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "dqn"],
        help="RL algorithm used for the model (ppo, a2c, or dqn; default: ppo)"
    )
    
    args = parser.parse_args()
    
    # Register environment
    register_race_env()
    
    # Load GP data
    try:
        gp_data = load_gp_from_json(args.gp)
        print(f"Loaded GP data from: {args.gp}")
    except Exception as e:
        print(f"Error loading GP data: {e}. Falling back to default/synthetic gp_data.")
        gp_data = None
        
    # Create environment
    render_mode = "human" if args.render else None
    env = RaceGymEnv(gp_data=gp_data, render_mode=render_mode)
    
    # Select the correct RL algorithm class
    algo_classes = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN
    }
    model_class = algo_classes[args.algo]
    
    # Load model
    model_zip = args.model
    if not model_zip.endswith(".zip"):
        model_zip_path = f"{model_zip}.zip"
    else:
        model_zip_path = model_zip
        # Strip extension for model class load
        model_zip = model_zip[:-4]
        
    print(f"Loading {args.algo.upper()} model from {model_zip_path}...")
    try:
        model = model_class.load(model_zip, env=env)
    except FileNotFoundError:
        print(f"Error: Model file '{model_zip_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Evaluate model
    print(f"Starting evaluation of {args.episodes} episodes...")
    evaluate_model(
        model=model,
        env=env,
        n_episodes=args.episodes,
        deterministic=not args.stochastic
    )
    
    env.close()


if __name__ == "__main__":
    main()
