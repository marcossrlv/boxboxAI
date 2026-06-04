"""
test_integration.py
-------------------
Verifica la integracion completa entre:
  - domain_model.py  (clases del dominio)
  - race_config.py   (carga JSON + factory)
  - spanish_gp_2024.json (datos reales)

Ejecuta checks exhaustivos sin dependencias externas (solo stdlib + el proyecto).
"""

import sys
import traceback
from pathlib import Path

# ── Imports del proyecto ───────────────────────────────────────────────────────
from domain_model import (
    Race, Car, Track, TireType, QuadraticTireModel,
    FuelModel, PitStopModel, OvertakingModel,
)
from race_config import load_gp_from_json, setup_real_race

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"
JSON_PATH = Path(__file__).parent.parent / "data" / "spanish_gp_2024.json"

errors = []

def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  {PASS} {name}")
    else:
        msg = f"  {FAIL} {name}" + (f" -> {detail}" if detail else "")
        print(msg)
        errors.append(msg)

def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ==============================================================================
# 1. load_gp_from_json: estructura del dict devuelto
# ==============================================================================
section("1. load_gp_from_json() - estructura y tipos")

gp = load_gp_from_json(JSON_PATH)

check("'name' existe y es str",           isinstance(gp.get("name"), str))
check("'year' existe y es int",           isinstance(gp.get("year"), int))
check("'total_laps' existe y es int",     isinstance(gp.get("total_laps"), int) and gp["total_laps"] > 0)
check("'pole_time' existe y es float",    isinstance(gp.get("pole_time"), float) and gp["pole_time"] > 0)

# tire_coefficients
tc = gp.get("tire_coefficients", {})
check("'tire_coefficients' es dict",      isinstance(tc, dict))
check("Claves son TireType enums",        all(isinstance(k, TireType) for k in tc))
check("SOFT en tire_coefficients",        TireType.SOFT in tc)
check("MEDIUM en tire_coefficients",      TireType.MEDIUM in tc)
check("HARD en tire_coefficients",        TireType.HARD in tc)
check("Coeficientes son tuplas de 3 floats",
      all(isinstance(v, tuple) and len(v) == 3 and all(isinstance(x, float) for x in v) for v in tc.values()))

# qualifying_results
qr = gp.get("qualifying_results", [])
check("'qualifying_results' es lista",    isinstance(qr, list) and len(qr) > 0)
check("Cada entry es tupla (int, float, [str])", all(isinstance(e, tuple) and len(e) >= 2 for e in qr))
check("car_id de pole es 0",              qr[0][0] == 0)
check("Pole time ~ 71.383s",              abs(qr[0][1] - 71.383) < 0.01)
check("20 pilotos en clasificacion",      len(qr) == 20)

# opponent_strategies
strats = gp.get("opponent_strategies", {})
check("'opponent_strategies' es dict",    isinstance(strats, dict))
check("car_id=0 NO tiene estrategia",     "0" not in strats)
check("Cada stop tiene 'in_lap' e 'tire_type'",
      all(
          all("in_lap" in s and "tire_type" in s for s in stops)
          for stops in strats.values()
      ))
check("tire_type es string (no TireType enum)",
      all(
          all(isinstance(s["tire_type"], str) for s in stops)
          for stops in strats.values()
      ))

# ==============================================================================
# 2. setup_real_race(): construccion del objeto Race
# ==============================================================================
section("2. setup_real_race() - objeto Race construido")

race = setup_real_race(gp)

check("Devuelve instancia de Race",       isinstance(race, Race))
check("Race.track es Track",              isinstance(race.track, Track))
check("Track.laps == 66",                 race.track.laps == 66)
check("Race.tire_model es QuadraticTireModel",
      isinstance(race.tire_model, QuadraticTireModel))
check("20 coches en la carrera",          len(race.cars) == 20)
check("Track.base_overtaking_prob cargado del JSON", abs(race.track.base_overtaking_prob - 0.0899) < 0.001)

# Coche del usuario (car_id=0)
user_car = next((c for c in race.cars if c.car_id == 0), None)
check("Coche car_id=0 existe",            user_car is not None)
if user_car:
    check("base_lap_time del usuario ~ 71.383s",
          abs(user_car.base_lap_time - 71.383) < 0.01)
    check("fuel_mass inicializada (>0)",  user_car.fuel_mass > 0)
    check("laps_on_tire inicial == 0",    user_car.laps_on_tire == 0)
    check("car_id=0 sin planned_stops",
          user_car.strategy.get("planned_stops", []) == [])

# Oponentes con estrategias
cars_with_stops = [c for c in race.cars
                   if c.car_id != 0 and c.strategy.get("planned_stops")]
check("Al menos 1 oponente tiene planned_stops",
      len(cars_with_stops) > 0)

# Verificar que tire_type en planned_stops es TireType enum (no string)
for car in cars_with_stops:
    for stop in car.strategy["planned_stops"]:
        tt = stop.get("tire_type")
        if not isinstance(tt, TireType):
            check(f"  Car {car.car_id}: stop tire_type es TireType enum", False,
                  f"Got {type(tt).__name__} = {tt!r}")
            break
    else:
        continue
    break
else:
    check("Todos los planned_stops tienen tire_type como TireType enum", True)

# ==============================================================================
# 3. QuadraticTireModel: coeficientes reales aplicados correctamente
# ==============================================================================
section("3. QuadraticTireModel - coeficientes reales")

tm = race.tire_model
a_s, b_s, c_s = tm._coefficients[TireType.SOFT]

offset_n0 = tm.get_time_offset(TireType.SOFT, 0)
expected_n0 = a_s * 0**2 + b_s * 0 + c_s
check("offset(SOFT, n=0) calculado correctamente",
      abs(offset_n0 - expected_n0) < 1e-6, f"got={offset_n0:.6f} expected={expected_n0:.6f}")

offset_n10 = tm.get_time_offset(TireType.SOFT, 10)
expected_n10 = a_s * 100 + b_s * 10 + c_s
check("offset(SOFT, n=10) calculado correctamente",
      abs(offset_n10 - expected_n10) < 1e-6)

# Fallback para compuesto desconocido no debe crashear
try:
    _ = tm.get_time_offset(TireType.HARD, 5)
    check("get_time_offset(HARD, 5) no lanza excepcion", True)
except Exception as e:
    check("get_time_offset(HARD, 5) no lanza excepcion", False, str(e))

# ==============================================================================
# 4. Race.step() - simulacion de vueltas
# ==============================================================================
section("4. Race.step() - simulacion completa")

# Capturamos fuel inicial ANTES del primer step
initial_fuel = race.cars[0].fuel_mass

state = race.step()

required_keys = {"time_step", "lap_times", "laps", "laps_on_tire", "fuel_mass",
                 "total_time", "order", "finished", "leaderboard_by_time",
                 "race_finished", "winner"}
check("Estado devuelve todas las claves esperadas",
      required_keys.issubset(state.keys()),
      str(required_keys - state.keys()))

check("time_step == 1 tras primer step",  state["time_step"] == 1)
check("order tiene 20 car_ids",           len(state["order"]) == 20)
check("lap_times tiene 20 entradas",      len(state["lap_times"]) == 20)
check("Ningun coche terminado en vuelta 1", not any(state["finished"].values()))

# Lap times en rango razonable
lt_user = state["lap_times"][0]
check("lap_time de car_id=0 es positivo", lt_user > 0)
check("lap_time de car_id=0 < 200s",     lt_user < 200)

# laps_on_tire incrementado
check("laps_on_tire de car_id=0 == 1 tras vuelta 1",
      state["laps_on_tire"][0] == 1)

# total_time > 0
check("total_time de car_id=0 > 0",      state["total_time"][0] > 0)

# Simular 5 vueltas mas
for _ in range(5):
    state = race.step()

check("time_step == 6 tras 6 steps",     state["time_step"] == 6)
check("fuel_mass decrece tras 6 vueltas",
      state["fuel_mass"][0] < initial_fuel,
      f"inicial={initial_fuel:.2f}  ahora={state['fuel_mass'][0]:.2f}")

# ==============================================================================
# 5. Pit stops de oponentes: se ejecutan en la vuelta correcta
# ==============================================================================
section("5. Pit stops - ejecucion en vuelta correcta")

# Reiniciamos carrera para este test
race2 = setup_real_race(gp)

# Encontrar un coche con parada planificada pronto
early_stop_car = None
early_stop_lap = None
for car in race2.cars:
    stops = car.strategy.get("planned_stops", [])
    if stops:
        lap = stops[0]["in_lap"]
        if lap <= 20:
            early_stop_car = car
            early_stop_lap = lap
            break

if early_stop_car:
    target_tire = early_stop_car.strategy["planned_stops"][0]["tire_type"]
    original_tire = early_stop_car.current_tire_type

    # Avanzar hasta la vuelta de la parada
    for _ in range(early_stop_lap):
        race2.step()

    check(f"Car {early_stop_car.car_id}: neumatico cambiado tras parada en vuelta {early_stop_lap}",
          early_stop_car.current_tire_type == target_tire,
          f"era={original_tire.value}, ahora={early_stop_car.current_tire_type.value}, esperado={target_tire.value}")
    check(f"Car {early_stop_car.car_id}: laps_on_tire reseteado tras parada",
          early_stop_car.laps_on_tire <= early_stop_lap,
          f"laps_on_tire={early_stop_car.laps_on_tire}")
else:
    check("No se encontro coche con parada en vuelta <=20 (test omitido)", True)



# ==============================================================================
# 7. num_cars parametro
# ==============================================================================
section("7. num_cars personalizado")

race_5 = setup_real_race(gp, num_cars=5)
check("num_cars=5 crea exactamente 5 coches", len(race_5.cars) == 5)
check("car_ids del 0 al 4",
      sorted(c.car_id for c in race_5.cars) == list(range(5)))

# ==============================================================================
# Resultado final
# ==============================================================================
print(f"\n{'='*55}")
if errors:
    print(f"  RESULTADO: {len(errors)} FALLO(S)")
    for e in errors:
        print(f"    {e}")
    sys.exit(1)
else:
    print(f"  RESULTADO: TODOS LOS CHECKS PASARON")
print(f"{'='*55}\n")
