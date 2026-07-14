"""
Evaluation script for trained DRL agents in Race Simulation
"""

import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from race_gym_env import RaceGymEnv, register_race_env
from race_config import load_gp_from_json
import os


def evaluate_model(model, env, n_episodes, deterministic=True, gp_data=None):
    """
    Evaluate a loaded model on a given environment using a Monte Carlo study on the stochastic environment.
    
    Args:
        model: The trained Stable-Baselines3 model.
        env: The Gymnasium environment.
        n_episodes: Number of episodes to run the evaluation (independent trials).
        deterministic: Whether to use deterministic actions (True for greedy policy).
        gp_data: Dict with GP configuration and real results.
        
    Returns:
        tuple: (episode_rewards, episode_lengths, episode_positions)
    """
    import matplotlib.pyplot as plt
    import collections
    import os
    
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
    print(f"Running Monte Carlo study with N = {n_episodes} independent simulations.")
    print(f"Agent Action Selection Mode: {'DETERMINISTIC (Greedy/Optimal)' if deterministic else 'STOCHASTIC (Exploratory)'}")
    print(f"Environment Mode: STOCHASTIC (Fluctuations in overtakes, lap times, tire degradation)")
    if agent_real:
        print(f"Comparing against real driver: {driver_name} (Started P{real_grid}, Finished P{real_pos})")
        if real_time_s:
            print(f"Real Race Time: {real_time_s:.3f}s")
        print(f"Real Strategy: {real_strategy_str}")
    print("=" * 80)
        
    for episode in range(n_episodes):
        # Unique seed for each episode to guarantee independent and reproducible Monte Carlo trials
        episode_seed = 1000 + episode
        
        # Seed both the gym environment, Python's random, and the race rng
        import random
        random.seed(episode_seed)
        obs, info = env.reset(seed=episode_seed)
        env.race.rng.seed(episode_seed)
        
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
        
        # Only print individual episode details if N is small, to avoid flooding the console
        if n_episodes <= 20:
            print(f"Eval Episode {episode + 1}: Position={episode_positions[-1]}, Time={agent_sim_time:.3f}s, Reward={total_reward:.2f}, Strategy: {strategy_str}")
        elif (episode + 1) % 10 == 0 or (episode + 1) == n_episodes:
            # Print periodic progress update for large N
            print(f"Completed {episode + 1}/{n_episodes} simulations...")
            
    # Calculate statistics
    avg_pos = np.mean(episode_positions)
    std_pos = np.std(episode_positions)
    avg_time = np.mean(episode_times) if episode_times else 0.0
    std_time = np.std(episode_times) if episode_times else 0.0
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    win_rate = (wins / n_episodes) * 100
    
    print("\n" + "="*95)
    print(f"--- MONTE CARLO STUDY RESULTS: SIMULATOR VS REAL ({driver_name}) ---")
    print(f"Total Runs (N): {n_episodes}")
    print(f"GP: {gp_data.get('name', 'Unknown')} | Real Driver: {driver_name}")
    print(f"Real Grid Position: P{real_grid if real_grid is not None else 'N/A'} | Real Final Position: P{real_pos if real_pos is not None else 'N/A'}")
    if real_time_s:
        print(f"Real Race Time: {real_time_s:.3f}s")
    print(f"Real Strategy: {real_strategy_str}")
    print("-"*95)
    
    # If N is small, print detailed row-by-row table
    if n_episodes <= 20:
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
        
    # Print aggregated statistics
    avg_pos_diff = f"{avg_pos - real_pos:+.1f}" if real_pos is not None else "N/A"
    if real_time_s and avg_time:
        avg_time_diff = f"{avg_time - real_time_s:+.3f}s"
        real_time_summary_str = f"{real_time_s:.3f}s"
    else:
        avg_time_diff = "N/A"
        real_time_summary_str = "N/A"
        
    print(f"Posición Final:           P{avg_pos:.2f} ± {std_pos:.2f} (Real: P{real_pos if real_pos is not None else 'N/A'}, Diff: {avg_pos_diff})")
    print(f"Tiempo de Carrera:         {avg_time:.3f}s ± {std_time:.2f}s (Real: {real_time_summary_str}, Diff: {avg_time_diff})")
    print(f"Recompensa (Reward):       {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Tasa de Victorias (Win %): {wins}/{n_episodes} ({win_rate:.1f}%)")
    print("="*95)
    
    # Generate and save the visualization plot
    try:
        plt.figure(figsize=(10, 6))
        
        if n_episodes > 20:
            # Distribution of final positions
            pos_counts = collections.Counter(episode_positions)
            positions_sorted = sorted(list(pos_counts.keys()))
            percentages = [pos_counts[p] / n_episodes * 100 for p in positions_sorted]
            
            bars = plt.bar(positions_sorted, percentages, color='#1f77b4', edgecolor='black', alpha=0.8, width=0.5)
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.xlabel("Final Position", fontsize=12)
            plt.ylabel("Percentage of Runs (%)", fontsize=12)
            plt.xticks(positions_sorted, [f"P{p}" for p in positions_sorted])
            plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        else:
            # Line plot of simulator positions
            plt.plot(range(1, n_episodes + 1), episode_positions, marker='o', color='#1f77b4', linestyle='--', label='Simulator Final Position')
            if real_pos is not None:
                plt.axhline(y=real_pos, color='#2ca02c', linestyle='-', linewidth=2.5, label=f'Real Final Position (P{real_pos} - {driver_name})')
            if real_grid is not None:
                plt.axhline(y=real_grid, color='#d62728', linestyle=':', linewidth=2, label=f'Starting Grid Position (P{real_grid})')
            plt.xlabel("Simulator Run (Episode)", fontsize=12)
            plt.ylabel("Final Position", fontsize=12)
            plt.xticks(range(1, n_episodes + 1))
            plt.gca().invert_yaxis()  # P1 at the top
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='best', frameon=True, shadow=True)
            
        gp_name = gp_data.get('name', 'Grand Prix') if gp_data else 'Grand Prix'
        gp_year = gp_data.get('year', '') if gp_data else ''
        plt.title(f"Monte Carlo Study: Agent Performance Distribution\n{gp_name} ({gp_year})", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "evaluation_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[OK] Comparison plot saved as '{plot_path}'")
    except Exception as e:
        print(f"Error generating plot: {e}")
        
    return episode_rewards, episode_lengths, episode_positions


def run_simulation(model, env, seed, strategy_type="agent", deterministic=True, gp_data=None):
    """
    Runs a single simulation episode.
    strategy_type can be:
      - "agent": Use the DRL model's policy
      - "real": Force pit stops based on the driver's real strategy from gp_data (or fallback)
      - "heuristic": Force pit stops at Lap 22 (MEDIUM) and Lap 44 (HARD)
    """
    import random
    random.seed(seed)
    obs, info = env.reset(seed=seed)
    env.race.rng.seed(seed)
    
    total_reward = 0
    steps = 0
    pit_stops = []
    
    # Track metrics per lap
    laps_data = []
    
    starting_compound = env.agent_car.current_tire_type.value.upper()
    
    while True:
        current_lap = env.agent_car.lap
        current_wear = env.agent_car.laps_on_tire
        current_compound = env.agent_car.current_tire_type.value.upper()
        
        # Determine action
        if strategy_type == "agent":
            action, _states = model.predict(obs, deterministic=deterministic)
            action_val = int(action.item()) if hasattr(action, "item") else int(action)
        elif strategy_type == "real":
            action_val = 0
            strategy = []
            if gp_data:
                real_results = gp_data.get("real_results", {})
                agent_real = real_results.get(str(env.agent_car_id), {})
                strategy = agent_real.get("strategy", [])
            
            if strategy:
                for stop in strategy:
                    if stop.get("in_lap") == current_lap:
                        compound = stop.get("tire_type").upper()
                        action_val = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}.get(compound, 0)
                        break
            else:
                # Fallback to Norris
                if current_lap == 23:
                    action_val = 2  # Pit for MEDIUM
                elif current_lap == 47:
                    action_val = 1  # Pit for SOFT
        elif strategy_type == "heuristic":
            total_laps = env.race.track.laps
            stint_len = total_laps // 3
            stop1_lap = stint_len
            stop2_lap = stint_len * 2
            
            if current_lap == stop1_lap:
                action_val = 2  # Pit for MEDIUM
            elif current_lap == stop2_lap:
                action_val = 3  # Pit for HARD
            else:
                action_val = 0  # No pit
        else:
            action_val = 0
            
        if action_val in (1, 2, 3):
            compounds = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}
            pit_stops.append((current_lap, compounds[action_val]))
            
        obs, reward, terminated, truncated, info = env.step(action_val)
        total_reward += reward
        steps += 1
        
        # Record lap-by-lap data AFTER step
        completed_lap = env.agent_car.lap - 1
        try:
            position = env.race.order.index(env.agent_car_id) + 1
        except ValueError:
            position = env.num_cars
            
        # Calculate gap to leader
        race_state = info.get('race_state', {})
        leaderboard = race_state.get('leaderboard_by_time', [])
        
        try:
            agent_time = next(t for cid, t in leaderboard if cid == env.agent_car_id)
        except StopIteration:
            agent_time = env.agent_car.total_time
            
        leader_time = leaderboard[0][1] if leaderboard else agent_time
        gap_to_leader = agent_time - leader_time
        
        # Get tire offset (degradation penalty in seconds)
        current_offset = env.race.tire_model.get_time_offset(env.agent_car.current_tire_type, env.agent_car.laps_on_tire)
            
        laps_data.append({
            'lap': completed_lap,
            'position': position,
            'wear': env.agent_car.laps_on_tire,
            'offset': current_offset,
            'gap': gap_to_leader,
            'compound': env.agent_car.current_tire_type.value.upper(),
            'lap_time': env.agent_car.current_lap_time
        })
        
        if terminated or truncated:
            try:
                final_rank = [car_id for car_id, _ in leaderboard].index(env.agent_car_id) + 1
                agent_sim_time = next(t for cid, t in leaderboard if cid == env.agent_car_id)
            except ValueError:
                final_rank = env.num_cars
                agent_sim_time = 9999.9
            
            winner_time = leaderboard[0][1] if leaderboard else 0.0
            break
            
    # Format strategy string
    strategy_str = starting_compound
    for lap, comp in pit_stops:
        strategy_str += f" -> [Lap {lap}: {comp}]"
        
    return {
        'final_position': final_rank,
        'total_time': agent_sim_time,
        'winner_time': winner_time,
        'total_reward': total_reward,
        'strategy': strategy_str,
        'laps_data': laps_data,
        'pit_stops': pit_stops
    }


def run_compare_suite(model, env, gp_data):
    """
    Runs the multi-scenario comparison suite and saves the results.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    real_results_data = gp_data.get("real_results", {})
    agent_real = real_results_data.get(str(env.agent_car_id))
    driver_name = "Real Driver"
    if agent_real:
        driver_name = agent_real.get("driver_name", "Real Driver")
        
    print("\n" + "=" * 90)
    print("      INICIANDO SUITE DE COMPARACIÓN DE ESTRATEGIAS (BARCELONA - 66 VUELTAS)")
    print("=" * 90)
    
    # 1. Baseline Humano
    print(f"\n[1/3] Simulando Escenario Real (Baseline Humano: {driver_name})...")
    real_results = run_simulation(model, env, seed=42, strategy_type="real", gp_data=gp_data)
    print(f"      Completado: Posición Final P{real_results['final_position']}, Tiempo = {real_results['total_time']:.3f}s")
    
    # 2. Línea Base Estática (Heurística)
    print("\n[2/3] Simulando Estrategia Heurística (Línea Base Estática)...")
    heuristic_results = run_simulation(model, env, seed=42, strategy_type="heuristic", gp_data=gp_data)
    print(f"      Completado: Posición Final P{heuristic_results['final_position']}, Tiempo = {heuristic_results['total_time']:.3f}s")
    
    # 3. Agente Inteligente (Determinista)
    print("\n[3/3] Simulando Agente DRL (Modo Determinista)...")
    agent_results = run_simulation(model, env, seed=42, strategy_type="agent", deterministic=True, gp_data=gp_data)
    print(f"      Completado: Posición Final P{agent_results['final_position']}, Tiempo = {agent_results['total_time']:.3f}s")
    
    # Imprimir la Tabla Maestra de Rendimiento con Métricas Absolutas
    t_agent = agent_results['total_time']
    t_real = real_results['total_time']
    t_heur = heuristic_results['total_time']
    
    delta_real_vs_real = "Baseline"
    delta_real_vs_heur = f"{t_real - t_heur:+.3f}s"
    
    delta_heur_vs_real = f"{t_heur - t_real:+.3f}s"
    delta_heur_vs_heur = "Baseline"
    
    delta_agent_vs_real = f"{t_agent - t_real:+.3f}s"
    delta_agent_vs_heur = f"{t_agent - t_heur:+.3f}s"
    
    print("\n" + "=" * 125)
    print("                                  ELEMENTO 1: TABLA MAESTRA DE RENDIMIENTO (MÉTRICAS ABSOLUTAS)")
    print("=" * 125)
    print(f"{'Escenario':<32} | {'Pos Final':<10} | {'Tiempo Total':<13} | {'Delta vs Real':<14} | {'Delta vs Heur':<14} | {'Estrategia de Paradas (Vuelta: Neumático)':<35}")
    print("-" * 125)
    print(f"{f'Baseline Humano ({driver_name} Real)':<32} | P{real_results['final_position']:<9} | {t_real:<11.3f}s | {delta_real_vs_real:<14} | {delta_real_vs_heur:<14} | {real_results['strategy']}")
    print(f"{'Línea Base Estática (Heurística)':<32} | P{heuristic_results['final_position']:<9} | {t_heur:<11.3f}s | {delta_heur_vs_real:<14} | {delta_heur_vs_heur:<14} | {heuristic_results['strategy']}")
    print(f"{'Agente Inteligente (DRL Det.)':<32} | P{agent_results['final_position']:<9} | {t_agent:<11.3f}s | {delta_agent_vs_real:<14} | {delta_agent_vs_heur:<14} | {agent_results['strategy']}")
    print("=" * 125 + "\n")
    
    # Generar gráficos
    try:
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Configurar estilo visual premium
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        
        gp_name = gp_data.get('name', 'Spanish Grand Prix')
        gp_year = gp_data.get('year', 2024)
        

        # --- ELEMENTO 3: GRÁFICO DE STINTS Y EVOLUCIÓN DEL DESGASTE DE NEUMÁTICOS ---
        plt.style.use('dark_background')
        fig_wear = plt.figure(figsize=(11, 6.5), facecolor='#111111')
        ax_wear = fig_wear.add_subplot(1, 1, 1, facecolor='#111111')
        
        # Grid and spines configuration for premium look
        ax_wear.grid(True, which='both', linestyle=':', color='#333333', alpha=0.5)
        for spine in ax_wear.spines.values():
            spine.set_color('#444444')
            spine.set_linewidth(1.2)
            
        def plot_wear_stints(ax, laps_data, linestyle, linewidth, label_prefix, alpha=1.0):
            stints = []
            current_stint = []
            for entry in laps_data:
                # Group by compound and ensure wear is increasing (resets on pit stops)
                if not current_stint or (entry['compound'] == current_stint[-1]['compound'] and entry['wear'] >= current_stint[-1]['wear']):
                    current_stint.append(entry)
                else:
                    stints.append(current_stint)
                    current_stint = [entry]
            if current_stint:
                stints.append(current_stint)
                
            compound_colors = {
                "SOFT": "#E10600",     # F1 Red
                "MEDIUM": "#FAD02C",   # F1 Yellow
                "HARD": "#FFFFFF"      # F1 White (requested by user)
            }
            
            total_laps = float(gp_data.get('total_laps', 66))
            
            for i, stint in enumerate(stints):
                stint_laps = [e['lap'] for e in stint]
                # Normalize wear between 0.0 and 1.0 (laps_on_tire / total_laps)
                stint_wears = [float(e['wear']) / total_laps for e in stint]
                comp = stint[0]['compound']
                
                # Prepend starting point for the stint (lap 0 at 0 wear, or previous pit lap at 0 wear)
                if i == 0:
                    stint_laps.insert(0, 0)
                    stint_wears.insert(0, 0.0)
                else:
                    prev_last_lap = stints[i-1][-1]['lap']
                    stint_laps.insert(0, prev_last_lap)
                    stint_wears.insert(0, 0.0)
                
                # Append vertical drop to 0 if not the last stint
                if i < len(stints) - 1:
                    last_lap = stint[-1]['lap']
                    stint_laps.append(last_lap)
                    stint_wears.append(0.0)
                
                ax.plot(stint_laps, stint_wears, color=compound_colors.get(comp, "#888888"), 
                        linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                        
        # Dibujamos la línea del piloto real (discontinua)
        plot_wear_stints(ax_wear, real_results['laps_data'], linestyle='--', linewidth=2.5, label_prefix='Real', alpha=0.6)
        # Dibujamos la línea heurística (punteada)
        plot_wear_stints(ax_wear, heuristic_results['laps_data'], linestyle=':', linewidth=2.0, label_prefix='Heur', alpha=0.5)
        # Dibujamos la línea del agente (sólida)
        plot_wear_stints(ax_wear, agent_results['laps_data'], linestyle='-', linewidth=2.5, label_prefix='Agent', alpha=1.0)
        
        ax_wear.set_title(f"Stints y Evolución del Desgaste de Neumáticos\nGP de {gp_name} {gp_year}", fontsize=14, fontweight='bold', pad=15, color='#FFFFFF')
        ax_wear.set_xlabel("Vuelta de Carrera", fontsize=11, color='#FFFFFF', labelpad=8)
        ax_wear.set_ylabel("Desgaste del Neumático (tire_wear: 0.0 a 1.0)", fontsize=11, color='#FFFFFF', labelpad=8)
        ax_wear.set_xlim(0, int(gp_data.get('total_laps', 66)))
        ax_wear.set_ylim(-0.02, 1.02)
        ax_wear.tick_params(colors='#AAAAAA', labelsize=10)
        
        legend_elements = [
            Line2D([0], [0], color='#AAAAAA', linestyle='-', linewidth=2.5, label='Agente DRL (Sólido)'),
            Line2D([0], [0], color='#AAAAAA', linestyle='--', linewidth=2.5, alpha=0.6, label=f'Piloto Real ({driver_name} - Guión)'),
            Line2D([0], [0], color='#AAAAAA', linestyle=':', linewidth=2.0, alpha=0.5, label='Línea Base Estática (Heurística - Puntos)'),
            Line2D([0], [0], color='#E10600', lw=4, label='SOFT (Rojo)'),
            Line2D([0], [0], color='#FAD02C', lw=4, label='MEDIUM (Amarillo)'),
            Line2D([0], [0], color='#FFFFFF', lw=4, label='HARD (Blanco)')
        ]
        ax_wear.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='#1E1E1E', edgecolor='#444444')
        plt.tight_layout()
        
        plot_path_wear = os.path.join(plot_dir, "tire_degradation.png")
        plt.savefig(plot_path_wear, dpi=300, facecolor='#111111')
        plt.close()
        print(f"[OK] Gráfico de ciclo de vida del neumático guardado en '{plot_path_wear}'")
        
    except Exception as e:
        print(f"Error generando los gráficos comparativos: {e}")
        import traceback
        traceback.print_exc()


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
        default=100, 
        help="Number of episodes to evaluate (default: 100)"
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the multi-scenario strategy comparison suite (Real, Heuristic, DRL, Robustness)"
    )
    
    args = parser.parse_args()
    
    # Register environment
    register_race_env()
    
    # Load GP data
    gp_data = load_gp_from_json(args.gp)
    print(f"Loaded GP data from: {args.gp}")
        
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
        
    # Evaluate model or run comparison suite
    if args.compare:
        run_compare_suite(
            model=model,
            env=env,
            gp_data=gp_data
        )
    else:
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
