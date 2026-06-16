"""
Evaluation script for trained DRL agents in Race Simulation
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from race_gym_env import RaceGymEnv, register_race_env
from race_config import load_gp_from_json


def evaluate_model(model, env, n_episodes, deterministic=True, gp_data=None):
    """
    Evaluate a loaded model on a given environment.
    
    Args:
        model: The trained Stable-Baselines3 model.
        env: The Gymnasium environment.
        n_episodes: Number of episodes to run the evaluation.
        deterministic: Whether to use deterministic actions.
        gp_data: Dict with GP configuration and real results.
        
    Returns:
        tuple: (episode_rewards, episode_lengths, episode_positions)
    """
    import matplotlib.pyplot as plt
    
    episode_rewards = []
    episode_lengths = []
    episode_positions = []
    episode_times = []
    episode_strategies = []
    wins = 0
    
    # Extract real results if available
    real_results = gp_data.get("real_results", {}) if gp_data else {}
    agent_real = real_results.get(str(env.agent_car_id)) if real_results else None
    
    driver_name = "Real Driver"
    real_pos = None
    real_grid = None
    real_time_s = None
    real_strategy_str = "Unknown"
    
    if agent_real:
        driver_name = agent_real.get("driver_name", "Real Driver")
        real_pos = agent_real.get("final_position")
        real_grid = agent_real.get("grid_position")
        real_time_s = agent_real.get("total_time_seconds")
        
        starting_comp = agent_real.get("starting_compound", "UNKNOWN")
        real_strategy_str = starting_comp
        for stop in agent_real.get("strategy", []):
            real_strategy_str += f" -> [Lap {stop['in_lap']}: {stop['tire_type']}]"
            
    print(f"\nEvaluating on GP: {gp_data.get('name', 'Unknown GP')} ({gp_data.get('year', '')})")
    if agent_real:
        print(f"Comparing against real driver: {driver_name} (Started P{real_grid}, Finished P{real_pos})")
        if real_time_s:
            print(f"Real Race Time: {real_time_s:.3f}s")
        print(f"Real Strategy: {real_strategy_str}")
        print("=" * 80)
        
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
                    agent_sim_time = next(t for cid, t in leaderboard if cid == env.agent_car_id)
                except ValueError:
                    rank = env.num_cars  # fallback
                    agent_sim_time = 9999.9
                    
                episode_positions.append(rank)
                episode_times.append(agent_sim_time)
                
                if info.get('winner') == env.agent_car_id:  # Agent won
                    wins += 1
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Format the strategy string
        strategy_str = starting_compound
        for lap, comp in pit_stops:
            strategy_str += f" -> [Lap {lap}: {comp}]"
        episode_strategies.append(strategy_str)
            
        print(f"Eval Episode {episode + 1}: Position={episode_positions[-1]}, Time={agent_sim_time:.3f}s, Reward={total_reward:.2f}, Strategy: {strategy_str}")
        
        # Print final leaderboard for this episode
        driver_names = {car.car_id: car.driver_name for car in env.race.cars}
        print(f"\n--- CLASIFICACIÓN FINAL (EPISODIO {episode + 1}) ---")
        print(f"{'Pos':>3} | {'Piloto':<25} | {'Car ID':>6} | {'Tiempo':>12}")
        print("-" * 55)
        
        winner_time = None
        for pos, (car_id, total_time) in enumerate(leaderboard):
            driver_name_curr = driver_names.get(car_id, f"Driver {car_id}")
            # Highlight agent car
            if car_id == env.agent_car_id:
                driver_name_curr = f"* {driver_name_curr}"
            else:
                driver_name_curr = f"  {driver_name_curr}"
                
            if pos == 0:
                winner_time = total_time
                time_str = f"{total_time:.3f}s"
            else:
                gap = total_time - winner_time
                time_str = f"+{gap:.3f}s"
            print(f"{pos + 1:>3} | {driver_name_curr:<25} | {car_id:>6} | {time_str:>12}")
        print("-" * 55 + "\n")
    
    print("\n" + "="*95)
    print(f"--- SUMMARY COMPARISON: SIMULATOR VS REAL ({driver_name}) ---")
    print(f"GP: {gp_data.get('name', 'Unknown')} | Real Driver: {driver_name}")
    print(f"Real Grid Position: P{real_grid if real_grid is not None else 'N/A'} | Real Final Position: P{real_pos if real_pos is not None else 'N/A'}")
    if real_time_s:
        print(f"Real Race Time: {real_time_s:.3f}s")
    print(f"Real Strategy: {real_strategy_str}")
    print("-"*95)
    print(f"{'Episodio':<10} | {'Pos Sim':<8} | {'Pos Real':<8} | {'Pos Diff':<8} | {'Time Sim':<11} | {'Time Real':<11} | {'Time Diff':<10} | {'Strategy Sim'}")
    print("-"*95)
    
    for ep in range(n_episodes):
        sim_p = episode_positions[ep]
        sim_t = episode_times[ep]
        strat = episode_strategies[ep]
        
        pos_diff_str = f"{sim_p - real_pos:+d}" if real_pos is not None else "N/A"
        
        if real_time_s and sim_t is not None:
            time_diff = sim_t - real_time_s
            time_diff_str = f"{time_diff:+.3f}s"
            real_time_str = f"{real_time_s:.3f}s"
            sim_time_str = f"{sim_t:.3f}s"
        else:
            time_diff_str = "N/A"
            real_time_str = "N/A"
            sim_time_str = f"{sim_t:.3f}s" if sim_t is not None else "N/A"
            
        print(f"{ep + 1:<10} | P{sim_p:<7} | P{real_pos if real_pos is not None else 'N/A':<7} | {pos_diff_str:<8} | {sim_time_str:<11} | {real_time_str:<11} | {time_diff_str:<10} | {strat}")
        
    print("-"*95)
    avg_pos = np.mean(episode_positions)
    avg_time = np.mean(episode_times) if episode_times else 0.0
    avg_reward = np.mean(episode_rewards)
    
    avg_pos_diff = f"{avg_pos - real_pos:+.1f}" if real_pos is not None else "N/A"
    
    if real_time_s and avg_time:
        avg_time_diff = f"{avg_time - real_time_s:+.3f}s"
        real_time_summary_str = f"{real_time_s:.3f}s"
    else:
        avg_time_diff = "N/A"
        real_time_summary_str = "N/A"
        
    print(f"{'PROMEDIO':<10} | P{avg_pos:<7.1f} | P{real_pos if real_pos is not None else 'N/A':<7} | {avg_pos_diff:<8} | {avg_time:.3f}s | {real_time_summary_str:<11} | {avg_time_diff:<10} | (Pos StDev: ±{np.std(episode_positions):.2f})")
    print("="*95)
    print(f"Average Reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Win Rate: {wins}/{n_episodes} ({wins/n_episodes*100:.1f}%)")
    
    # Generate and save the visualization plot
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot simulated positions
        plt.plot(range(1, n_episodes + 1), episode_positions, marker='o', color='#1f77b4', linestyle='--', label='Simulator Final Position')
        
        # Plot real final position
        if real_pos is not None:
            plt.axhline(y=real_pos, color='#2ca02c', linestyle='-', linewidth=2.5, label=f'Real Final Position (P{real_pos} - {driver_name})')
            
        # Plot starting grid position
        if real_grid is not None:
            plt.axhline(y=real_grid, color='#d62728', linestyle=':', linewidth=2, label=f'Starting Grid Position (P{real_grid})')
            
        gp_name = gp_data.get('name', 'Grand Prix') if gp_data else 'Grand Prix'
        gp_year = gp_data.get('year', '') if gp_data else ''
        plt.title(f"Comparison of Agent Simulator vs Real Performance\n{gp_name} ({gp_year})", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulator Run (Episode)", fontsize=12)
        plt.ylabel("Final Position", fontsize=12)
        
        # Custom ticks
        plt.xticks(range(1, n_episodes + 1))
        all_positions = episode_positions + ([real_pos] if real_pos is not None else []) + ([real_grid] if real_grid is not None else [])
        min_pos = max(1, min(all_positions) - 1)
        max_pos = min(20, max(all_positions) + 1)
        plt.yticks(range(min_pos, max_pos + 1))
        
        plt.gca().invert_yaxis()  # P1 at the top
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best', frameon=True, shadow=True)
        plt.tight_layout()
        
        plot_path = "evaluation_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[OK] Comparison plot saved as '{plot_path}'")
    except Exception as e:
        print(f"Error generating plot: {e}")
    
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
        deterministic=not args.stochastic,
        gp_data=gp_data
    )
    
    env.close()


if __name__ == "__main__":
    main()
