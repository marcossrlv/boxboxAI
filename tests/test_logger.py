"""Prueba end-to-end del sistema de logging con una carrera completa."""
from race_config import load_gp_from_json
from race_gym_env import RaceGymEnv
from simulation_logger import SimulationLogger

gp = load_gp_from_json("spanish_gp_2024.json")
logger = SimulationLogger(gp_name="Spanish GP 2024", run_id="test_run")
env = RaceGymEnv(gp_data=gp, logger=logger)

# Episodio 1: no pit en ninguna vuelta
obs, _ = env.reset()
print("=== Episodio 1: corriendo carrera completa ===")
terminated = truncated = False
laps_done = 0
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(0)  # action=0: no pit
    laps_done += 1

print(f"Vueltas completadas: {laps_done}")
print(f"Ganador: {info['winner']}")

env.close()

# Verificar archivos generados
from pathlib import Path
run_dir = Path("logs/test_run")
laps_csv = run_dir / "laps.csv"
summary_json = run_dir / "summary.json"

print(f"\nArchivos generados:")
print(f"  laps.csv:     {laps_csv.stat().st_size:,} bytes  ({sum(1 for _ in open(laps_csv))-1} filas)")
print(f"  summary.json: {summary_json.stat().st_size:,} bytes")

import json
summary = json.loads(summary_json.read_text())
ep = summary[0]
print(f"\nResumen episodio 1:")
print(f"  GP:               {ep['gp_name']}")
print(f"  Total vueltas:    {ep['total_laps']}")
print(f"  Ganador car_id:   {ep['winner_car_id']}")
print(f"  Agente posicion:  {ep['agent']['final_position']}")
print(f"  Agente pit stops: {ep['agent']['pit_stops']}")
print(f"  Agente mejor lap: {ep['agent']['best_lap_time']}s")
print(f"  Clasificacion (top 5):")
for car_id, total in ep['leaderboard'][:5]:
    print(f"    Car {car_id:>2}: {total:.3f}s")
