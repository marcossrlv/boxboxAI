from race_config import load_gp_from_json
from race_gym_env import RaceGymEnv

gp = load_gp_from_json("data/spanish_gp_2024.json")
env = RaceGymEnv(gp_data=gp)

print(f"Coches en carrera:             {env.num_cars}")
print(f"Competidores (obs space):      {env.num_competitors}")
print(f"competitors_info shape:        {env.observation_space['competitors_info'].shape}")
print(f"Track laps:                    {env.race.track.laps}")

obs, info = env.reset()
print("\nObservation keys:", list(obs.keys()))
print("competitors_info shape en obs:", obs["competitors_info"].shape)

for i in range(3):
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Step {i+1}: reward={reward:.2f}  lap={env.agent_car.lap}  tire_wear={obs['tire_wear'][0]:.3f}")

print("\n[OK] RaceGymEnv + JSON integrado correctamente")
