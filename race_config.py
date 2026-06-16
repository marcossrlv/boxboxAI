"""
race_config.py
──────────────
Configura instancias de Race a partir de datos de Gran Premio.

Fuentes de datos:
  1. Archivo JSON generado por fetch_race_data.py  → load_gp_from_json()

Una vez obtenido el dict de datos, setup_real_race() construye la carrera.
"""

import json
from pathlib import Path

from domain_model import Race, Car, Track, TireType, QuadraticTireModel


# ─── Mapa string → TireType ───────────────────────────────────────────────────
_TIRE_MAP = {
    "SOFT": TireType.SOFT,
    "MEDIUM": TireType.MEDIUM,
    "HARD": TireType.HARD,
}


# ─── Carga desde JSON ─────────────────────────────────────────────────────────

def load_gp_from_json(json_path: str | Path) -> dict:
    """
    Carga un archivo JSON generado por fetch_race_data.py y lo convierte al
    formato interno del simulador (TireType como claves, tuplas para coeficientes).

    Args:
        json_path: Ruta al archivo JSON.

    Returns:
        Dict compatible con setup_real_race().
    """
    path = Path(json_path)
    if not path.exists():
        # Intentar buscar en la carpeta 'data/' relativa al directorio de race_config.py
        alt_path = Path(__file__).parent / "data" / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"No se encontró el archivo de datos: {path} (tampoco en {alt_path})")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Convertir coeficientes: keys string → TireType, listas → tuplas
    tire_coefficients = {}
    for compound_str, coeffs in raw["tire_coefficients"].items():
        tire_type = _TIRE_MAP.get(compound_str.upper())
        if tire_type is None:
            continue
        tire_coefficients[tire_type] = tuple(coeffs)

    # Convertir qualifying_results: lista de listas → lista de tuplas
    qualifying_results = [tuple(pair) for pair in raw["qualifying_results"]]

    return {
        "name": raw["name"],
        "year": raw.get("year"),
        "total_laps": raw["total_laps"],
        "pole_time": raw.get("pole_time"),
        "tire_coefficients": tire_coefficients,
        "qualifying_results": qualifying_results,
        "opponent_strategies": raw.get("opponent_strategies", {}),
        "real_results": raw.get("real_results", {}),
        "base_overtaking_prob": raw.get("base_overtaking_prob"),
    }


# ─── Factory ──────────────────────────────────────────────────────────────────

def setup_real_race(data: dict, num_cars: int = None) -> Race:
    """
    Construye una instancia de Race completa a partir del dict de datos del GP.

    Args:
        data:      Dict con datos del GP (de load_gp_from_json).
        num_cars:  Nº de coches a incluir. Si None, usa todos los de clasificación.

    Returns:
        Race lista para simular.
    """
    track = Track(
        laps=data["total_laps"],
        base_lap_time=0.0,
        name=data.get("name", ""),
        base_overtaking_prob=data.get("base_overtaking_prob")
    )

    # Los coeficientes pueden tener TireType o strings como claves
    raw_coeffs = data["tire_coefficients"]
    coefficients = {}
    
    defaults = {
        TireType.SOFT: (0.005, 0.03, 0.0),
        TireType.MEDIUM: (0.002, 0.015, 0.6),
        TireType.HARD: (0.001, 0.008, 1.2),
    }

    for key, val in raw_coeffs.items():
        original_key = key
        if isinstance(key, str):
            key = _TIRE_MAP.get(key.upper(), key)
            
        a, b, c = val
        # Si 'b' es muy negativo, significa que la curva está viciada por el efecto del 
        # vaciado de combustible en los datos reales. Usamos valores por defecto realistas.
        if b < -0.05:
            coefficients[key] = defaults.get(key, (0.002, 0.015, 0.5))
        else:
            coefficients[key] = tuple(val) if not isinstance(val, tuple) else val

    tire_model = QuadraticTireModel(coefficients=coefficients)

    qualy_data = data["qualifying_results"]
    strategies = data.get("opponent_strategies", {})
    limit = num_cars if num_cars else len(qualy_data)

    cars = []
    for i in range(limit):
        if i < len(qualy_data):
            # qualy_data[i] puede tener 2 o 3 elementos
            if len(qualy_data[i]) >= 3:
                car_id, q_time, driver_name = qualy_data[i][:3]
            else:
                car_id, q_time = qualy_data[i]
                driver_name = f"Driver {car_id}"
        else:
            last_time = qualy_data[-1][1] if qualy_data else 72.0
            car_id = i
            q_time = round(last_time + 0.1 * (i - len(qualy_data) + 1), 3)
            driver_name = f"Driver {car_id}"

        initial_tire = TireType.SOFT if i % 2 == 0 else TireType.MEDIUM

        car = Car(
            car_id=car_id,
            initial_tire_type=initial_tire,
            base_lap_time=q_time,
            driver_name=driver_name,
        )

        # Asignar estrategia del oponente si existe
        if car_id != 0:
            strat_key = str(car_id)
            if strat_key in strategies:
                planned_stops = []
                for stop in strategies[strat_key]:
                    tire_type = _TIRE_MAP.get(stop["tire_type"].upper())
                    if tire_type:
                        planned_stops.append({"in_lap": stop["in_lap"], "tire_type": tire_type})
                car.strategy["planned_stops"] = planned_stops

        cars.append(car)

    return Race(track, cars, tire_model=tire_model)


# ─── Prueba rápida ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Si se pasa un JSON como argumento, lo cargamos; si no, usamos el JSON por defecto
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "spanish_gp_2024.json"

    print(f"Cargando datos desde: {json_file}")
    gp_data = load_gp_from_json(json_file)

    race = setup_real_race(gp_data)
    print(f"\nCarrera: {gp_data['name']} - {race.track.laps} vueltas")
    print(f"{'P':>3} {'Driver':>20} {'Car ID':>6} {'Base Time':>10} {'Tire':>8}")
    print("-" * 52)
    for i, car in enumerate(race.cars):
        print(f"{i+1:>3} {car.driver_name:>20} {car.car_id:>6} {car.base_lap_time:>10.3f} {car.current_tire_type.value:>8}")

    print("\nSimulando 3 vueltas...")
    for _ in range(3):
        state = race.step()
        lap = state["time_step"]
        print(f"\n-- Vuelta {lap} --")
        for car_id in state["order"]:
            lt = state["lap_times"][car_id]
            tot = state["total_time"][car_id]
            age = state["laps_on_tire"][car_id]
            print(f"  Car {car_id:>2}: lap={lt:.3f}s  total={tot:.2f}s  tire_age={age}")
