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
            if current_lap == 22:
                action_val = 2  # Pit for MEDIUM
            elif current_lap == 44:
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
    import collections
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
    print(f"\n[1/4] Simulando Escenario Real (Baseline Humano: {driver_name})...")
    real_results = run_simulation(model, env, seed=42, strategy_type="real", gp_data=gp_data)
    print(f"      Completado: Posición Final P{real_results['final_position']}, Tiempo = {real_results['total_time']:.3f}s")
    
    # 2. Línea Base Estática (Heurística)
    print("\n[2/4] Simulando Estrategia Heurística (Línea Base Estática)...")
    heuristic_results = run_simulation(model, env, seed=42, strategy_type="heuristic", gp_data=gp_data)
    print(f"      Completado: Posición Final P{heuristic_results['final_position']}, Tiempo = {heuristic_results['total_time']:.3f}s")
    
    # 3. Agente Inteligente (Determinista)
    print("\n[3/4] Simulando Agente DRL (Modo Determinista)...")
    agent_results = run_simulation(model, env, seed=42, strategy_type="agent", deterministic=True, gp_data=gp_data)
    print(f"      Completado: Posición Final P{agent_results['final_position']}, Tiempo = {agent_results['total_time']:.3f}s")
    
    # 4. Prueba de Robustez (Estocástica)
    print("\n[4/4] Simulando Prueba de Robustez (100 carreras estocásticas)...")
    robustness_positions = []
    robustness_times = []
    robustness_strategies = []
    robustness_wins = 0
    
    for i in range(100):
        seed = 100 + i
        res = run_simulation(model, env, seed=seed, strategy_type="agent", deterministic=True, gp_data=gp_data)
        robustness_positions.append(res['final_position'])
        robustness_times.append(res['total_time'])
        robustness_strategies.append(res['strategy'])
        if res['final_position'] == 1:
            robustness_wins += 1
            
    avg_pos = np.mean(robustness_positions)
    std_pos = np.std(robustness_positions)
    avg_time = np.mean(robustness_times)
    std_time = np.std(robustness_times)
    win_rate = (robustness_wins / 100) * 100
    
    print("      Completado.")
    
    # Imprimir la Tabla Maestra de Rendimiento
    real_winner_time = real_results['winner_time']
    real_gap = real_results['total_time'] - real_winner_time
    real_gap_str = f"+{real_gap:.3f}s" if real_gap > 0 else "Ganador"
    
    heur_winner_time = heuristic_results['winner_time']
    heur_gap = heuristic_results['total_time'] - heur_winner_time
    heur_gap_str = f"+{heur_gap:.3f}s" if heur_gap > 0 else "Ganador"
    
    agent_winner_time = agent_results['winner_time']
    agent_gap = agent_results['total_time'] - agent_winner_time
    agent_gap_str = f"+{agent_gap:.3f}s" if agent_gap > 0 else "Ganador"
    
    print("\n" + "=" * 115)
    print("                                  ELEMENTO 1: TABLA MAESTRA DE RENDIMIENTO")
    print("=" * 115)
    print(f"{'Escenario':<32} | {'Pos Final':<10} | {'Tiempo Total':<13} | {'Dif. Ganador':<13} | {'Estrategia de Paradas (Vuelta: Neumático)':<35}")
    print("-" * 115)
    print(f"{f'Baseline Humano ({driver_name} Real)':<32} | P{real_results['final_position']:<9} | {real_results['total_time']:<11.3f}s | {real_gap_str:<13} | {real_results['strategy']}")
    print(f"{'Línea Base Estática (Heurística)':<32} | P{heuristic_results['final_position']:<9} | {heuristic_results['total_time']:<11.3f}s | {heur_gap_str:<13} | {heuristic_results['strategy']}")
    print(f"{'Agente Inteligente (DRL Det.)':<32} | P{agent_results['final_position']:<9} | {agent_results['total_time']:<11.3f}s | {agent_gap_str:<13} | {agent_results['strategy']}")
    print("-" * 115)
    print("Prueba de Robustez (Agente DRL - 100 carreras estocásticas):")
    print(f"  - Posición Final Promedio:    P{avg_pos:.2f} ± {std_pos:.2f} (Rango: P{min(robustness_positions)} - P{max(robustness_positions)})")
    print(f"  - Tiempo de Carrera Promedio:  {avg_time:.3f}s ± {std_time:.2f}s")
    print(f"  - Tasa de Victorias (P1):      {robustness_wins}/100 ({win_rate:.1f}%)")
    
    # Imprimir top estrategias en consola
    counter = collections.Counter(robustness_strategies)
    print("  - Estrategias más frecuentes:")
    for strat, count in counter.most_common(3):
        print(f"    * {strat:<60} | {count}% de uso")
    print("=" * 115 + "\n")
    
    # Generar gráficos
    try:
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Configurar estilo visual premium
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        
        laps = range(1, 67)
        gp_name = gp_data.get('name', 'Spanish Grand Prix')
        gp_year = gp_data.get('year', 2024)
        

        # --- ELEMENTO 3: GRÁFICO DE CICLO DE VIDA DEL NEUMÁTICO (DEGRADACIÓN EN S) ---
        fig_wear = plt.figure(figsize=(10, 6))
        ax_wear = fig_wear.add_subplot(1, 1, 1)
        
        def plot_wear_stints(ax, laps_data, linestyle, linewidth, label_prefix, alpha=1.0):
            stints = []
            current_stint = []
            for entry in laps_data:
                if not current_stint or entry['compound'] == current_stint[-1]['compound']:
                    current_stint.append(entry)
                else:
                    stints.append(current_stint)
                    current_stint = [entry]
            if current_stint:
                stints.append(current_stint)
                
            compound_colors = {
                "SOFT": "#E10600",
                "MEDIUM": "#FAD02C",
                "HARD": "#777777"
            }
            
            for stint in stints:
                stint_laps = [e['lap'] for e in stint]
                stint_offsets = [e['offset'] for e in stint]
                comp = stint[0]['compound']
                
                ax.plot(stint_laps, stint_offsets, color=compound_colors.get(comp, "#000000"), 
                        linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                        
        # Dibujamos primero la línea del piloto real (más gruesa y discontinua en el fondo)
        plot_wear_stints(ax_wear, real_results['laps_data'], linestyle='--', linewidth=4.0, label_prefix='Real', alpha=0.7)
        # Dibujamos encima la línea del agente (más fina y continua en primer plano)
        plot_wear_stints(ax_wear, agent_results['laps_data'], linestyle='-', linewidth=2.0, label_prefix='Agent', alpha=1.0)
        
        ax_wear.set_title(f"Ciclo de Vida del Neumático - {gp_name} {gp_year}", fontsize=13, fontweight='bold', pad=12)
        ax_wear.set_xlabel("Vuelta", fontsize=10)
        ax_wear.set_ylabel("Penalización por Desgaste (segundos / vuelta)", fontsize=10)
        ax_wear.set_xlim(1, 66)
        max_offset = max(max(d['offset'] for d in agent_results['laps_data']), max(d['offset'] for d in real_results['laps_data']))
        ax_wear.set_ylim(0, max_offset + 0.5)
        ax_wear.grid(True, linestyle=':', alpha=0.6)
        
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='-', linewidth=2.0, label='Agente DRL'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=4.0, alpha=0.7, label=f'Piloto Real ({driver_name})'),
            Line2D([0], [0], color='#E10600', lw=4, label='SOFT (Rojo)'),
            Line2D([0], [0], color='#FAD02C', lw=4, label='MEDIUM (Amarillo)'),
            Line2D([0], [0], color='#777777', lw=4, label='HARD (Gris)')
        ]
        ax_wear.legend(handles=legend_elements, loc='upper left', frameon=True)
        plt.tight_layout()
        
        plot_path_wear = os.path.join(plot_dir, "tire_degradation.png")
        plt.savefig(plot_path_wear, dpi=300)
        plt.close()
        print(f"[OK] Gráfico de ciclo de vida del neumático guardado en '{plot_path_wear}'")

        # --- ELEMENTO 4: ANÁLISIS DE ROBUSTEZ (HISTOGRAMA DE PORCENTAJES) ---
        fig_dist = plt.figure(figsize=(10, 6))
        ax_dist = fig_dist.add_subplot(1, 1, 1)
        
        pos_counts = collections.Counter(robustness_positions)
        positions = sorted(list(pos_counts.keys()))
        counts = [pos_counts[p] for p in positions]
        
        bars = ax_dist.bar(positions, counts, color='#2EC4B6', edgecolor='black', alpha=0.8, width=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax_dist.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
        ax_dist.set_title(f"Robustez del Agente DRL (100 carreras) - {gp_name} {gp_year}", fontsize=13, fontweight='bold', pad=12)
        ax_dist.set_xlabel("Posición Final", fontsize=10)
        ax_dist.set_ylabel("Porcentaje de Carreras (%)", fontsize=10)
        ax_dist.set_xticks(positions)
        ax_dist.set_xticklabels([f"P{p}" for p in positions])
        ax_dist.set_ylim(0, max(counts) * 1.15)
        ax_dist.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        summary_text = (
            f"Media: P{avg_pos:.2f} ± {std_pos:.2f}\n"
            f"Mejor: P{min(robustness_positions)}\n"
            f"Peor: P{max(robustness_positions)}\n"
            f"Tasa Victoria: {win_rate:.1f}%"
        )
        ax_dist.text(0.95, 0.95, summary_text, transform=ax_dist.transAxes, 
                    fontsize=10, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        plt.tight_layout()
        
        plot_path_dist = os.path.join(plot_dir, "robustness_distribution.png")
        plt.savefig(plot_path_dist, dpi=300)
        plt.close()
        print(f"[OK] Gráfico de robustez guardado en '{plot_path_dist}'\n")
        
        # --- ELEMENTO 5: TOP ESTRATEGIAS DEL AGENTE ---
        fig_strat = plt.figure(figsize=(10, 5))
        ax_strat = fig_strat.add_subplot(1, 1, 1)
        
        counter = collections.Counter(robustness_strategies)
        top_strats = counter.most_common(5)
        
        labels = [item[0] for item in top_strats]
        percentages = [(item[1] / len(robustness_strategies)) * 100 for item in top_strats]
        
        labels.reverse()
        percentages.reverse()
        
        bars = ax_strat.barh(labels, percentages, color='#2EC4B6', edgecolor='black', height=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax_strat.text(width + 1.0, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                          va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')
                          
        ax_strat.set_title(f"Top Estrategias Elegidas por el Agente (100 Carreras)\nGP {gp_name} {gp_year}", 
                           fontsize=13, fontweight='bold', pad=15)
        ax_strat.set_xlabel("Porcentaje de Uso (%)", fontsize=10)
        ax_strat.set_xlim(0, max(percentages) + 12.0 if percentages else 100)
        ax_strat.grid(True, axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        plot_path_strat = os.path.join(plot_dir, "agent_top_strategies.png")
        plt.savefig(plot_path_strat, dpi=300)
        plt.close()
        print(f"[OK] Gráfico de top estrategias guardado en '{plot_path_strat}'\n")
        
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
