"""
fetch_race_data.py
──────────────────
Extrae datos de un Gran Premio real usando FastF1 y genera un archivo JSON
con todo lo necesario para la simulación:

  - Nombre del circuito y número de vueltas
  - Resultados de clasificación (posición → tiempo en segundos)
  - Coeficientes cuadráticos (a, b, c) por compuesto, ajustados con numpy.polyfit
    sobre los tiempos por vuelta registrados en carrera

Uso:
    python fetch_race_data.py --year 2024 --gp "Spanish" --output spanish_gp_2024.json
    python fetch_race_data.py --year 2024 --gp "Spanish"  # usa nombre por defecto
"""

import argparse
import json
import sys
from pathlib import Path

import fastf1
import numpy as np


# ─── Configuración de caché de FastF1 ────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".fastf1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

COMPOUND_MAP = {
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
}

# Compuestos que queremos modelar (ignoramos INTER, WET, etc.)
TARGET_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def load_session(year: int, gp: str, session_type: str):
    print(f"[FastF1] Cargando sesión: {year} {gp} – {session_type} …")
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def get_qualifying_results(quali_session) -> tuple[list[tuple[int, float, str]], dict[str, int]]:
    """
    Devuelve lista de (car_id, lap_time_seconds, driver_name) ordenada por posición de clasificación,
    y un diccionario que mapea driver_number -> car_id.
    car_id = posición de clasificación - 1 (0-indexed, el poleman es car_id=0)
    """
    results = []
    driver_num_to_car_id = {}
    laps = quali_session.laps
    driver_info = {str(row["DriverNumber"]): row["FullName"] for _, row in quali_session.results.iterrows()}

    # Mejor vuelta por piloto: agrupar por DriverNumber y tomar el mínimo LapTime
    valid_laps = laps.dropna(subset=["LapTime"]).copy()
    idx_best = valid_laps.groupby("DriverNumber")["LapTime"].idxmin()
    best_laps = valid_laps.loc[idx_best].sort_values("LapTime")

    for grid_pos, (_, row) in enumerate(best_laps.iterrows()):
        lap_time_s = row["LapTime"].total_seconds()
        driver_num = str(row["DriverNumber"])
        if np.isnan(lap_time_s):
            continue
        driver_name = driver_info.get(driver_num, f"Driver {driver_num}")
        results.append((grid_pos, round(lap_time_s, 3), driver_name))
        driver_num_to_car_id[driver_num] = grid_pos

    return results, driver_num_to_car_id


def get_stint_data(race_session) -> dict[str, list[tuple[int, float]]]:
    """
    Para cada compuesto, devuelve una lista de (lap_number_in_stint, lap_time_seconds)
    filtrada: solo vueltas limpias (no laps tras safety car, etc.)
    """
    laps = race_session.laps

    # Filtrar vueltas: sin pit-out/pit-in, sin SC, tiempo válido
    # FastF1 v3.x usa IsOutLap / IsInLap en lugar de PitOutLap
    clean = laps.copy()
    for col in ("IsOutLap", "IsInLap"):
        if col in clean.columns:
            clean = clean[clean[col].fillna(False) == False]  # noqa: E712
    clean = clean[clean["TrackStatus"].isin(["1", "2"])]  # 1=Green, 2=Yellow suave
    clean = clean.dropna(subset=["LapTime", "Compound", "TyreLife"])

    stint_data: dict[str, list[tuple[int, float]]] = {c: [] for c in TARGET_COMPOUNDS}

    for _, row in clean.iterrows():
        compound = str(row["Compound"]).upper()
        if compound not in TARGET_COMPOUNDS:
            continue
        tyre_life = int(row["TyreLife"])     # vueltas de vida del neumático en ese momento
        lap_time_s = row["LapTime"].total_seconds()
        if np.isnan(lap_time_s) or lap_time_s <= 0:
            continue
        stint_data[compound].append((tyre_life, lap_time_s))

    return stint_data


def fit_quadratic(compound: str, data: list[tuple[int, float]], pole_time: float) -> tuple[float, float, float]:
    """
    Ajusta una parábola offset(n) = a·n² + b·n + c a los datos del compuesto.
    El offset es el tiempo extra respecto al tiempo pole (base del coche más rápido).

    Returns:
        (a, b, c) redondeados a 6 decimales
    """
    if len(data) < 3:
        print(f"  [AVISO] Compuesto {compound}: datos insuficientes ({len(data)} puntos). Usando defaults.")
        defaults = {"SOFT": (0.005, 0.03, 0.0), "MEDIUM": (0.002, 0.015, 0.6), "HARD": (0.001, 0.008, 1.2)}
        return defaults.get(compound, (0.002, 0.015, 0.5))

    ns = np.array([d[0] for d in data], dtype=float)
    times = np.array([d[1] for d in data], dtype=float)

    # Offset respecto al tiempo base (pole)
    offsets = times - pole_time

    # Ajuste de grado 2 con numpy.polyfit (mínimos cuadrados)
    coeffs = np.polyfit(ns, offsets, deg=2)
    a, b, c = coeffs

    print(f"  {compound}: a={a:.6f}  b={b:.6f}  c={c:.6f}  ({len(data)} puntos)")
    return (round(float(a), 6), round(float(b), 6), round(float(c), 6))


def calculate_overtaking_probability(race_session, default_prob: float = 0.08) -> float:
    """
    Calcula la probabilidad de adelantamiento empírica en el circuito de la sesión.
    Usa la relación entre adelantamientos reales y oportunidades en bandera verde (brecha <= 1.0s).
    Aplica un factor de escala y un rango límite [0.02, 0.18], ya que luego se aplican otros modificadores en la simulación.
    """
    try:
        laps = race_session.laps.copy()
        if laps.empty or "Position" not in laps.columns or "LapNumber" not in laps.columns:
            return default_prob

        # Limpiar datos nulos
        laps = laps.dropna(subset=["Position", "LapNumber", "Driver"])
        laps["LapNumber"] = laps["LapNumber"].astype(int)

        # Identificar paradas en boxes para cada piloto en cada vuelta
        laps["Pitted"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
        pitted_map = laps.set_index(["LapNumber", "Driver"])["Pitted"].to_dict()

        # Determinar el estado de pista por vuelta
        track_status = laps.groupby("LapNumber")["TrackStatus"].first().to_dict()

        # Tabla pivote de posiciones
        positions = laps.pivot(index="LapNumber", columns="Driver", values="Position")
        drivers = list(positions.columns)
        num_laps = int(positions.index.max())

        overtakes = 0
        for lap in range(2, num_laps + 1):
            status = track_status.get(lap, "1")
            # Ignorar vueltas bajo coche de seguridad o bandera roja
            # Status: 1=Verde, 2=Amarilla leve. Ignorar 4=SC, 5=Red, 6=VSC
            if any(c in status for c in ["4", "5", "6", "7"]):
                continue

            for i in range(len(drivers)):
                for j in range(i + 1, len(drivers)):
                    d1, d2 = drivers[i], drivers[j]
                    pos1_prev, pos2_prev = positions.at[lap-1, d1], positions.at[lap-1, d2]
                    pos1_curr, pos2_curr = positions.at[lap, d1], positions.at[lap, d2]
                    
                    if np.isnan(pos1_prev) or np.isnan(pos2_prev) or np.isnan(pos1_curr) or np.isnan(pos2_curr):
                        continue

                    # Verificar si hubo cruce/intercambio de posiciones
                    swapped = False
                    if pos1_prev > pos2_prev and pos1_curr < pos2_curr:
                        passer, passed = d1, d2
                        swapped = True
                    elif pos2_prev > pos1_prev and pos2_curr < pos1_curr:
                        passer, passed = d2, d1
                        swapped = True

                    if swapped:
                        # Excluir si alguno de los dos pilotos involucrados estaba en boxes esa vuelta
                        pitted_passer = pitted_map.get((lap, passer), False)
                        pitted_passed = pitted_map.get((lap, passed), False)
                        if not pitted_passer and not pitted_passed:
                            overtakes += 1

        # Calcular oportunidades de adelantamiento
        opportunities = 0
        for lap in range(1, num_laps):
            status = track_status.get(lap + 1, "1")
            if any(c in status for c in ["4", "5", "6", "7"]):
                continue

            # Obtener datos de la vuelta ordenados por el tiempo de paso por meta
            lap_data = laps[laps["LapNumber"] == lap].dropna(subset=["Time"])
            if lap_data.empty:
                continue

            lap_data_sorted = lap_data.sort_values("Time")
            times = lap_data_sorted["Time"].dt.total_seconds().values

            # Las diferencias entre coches consecutivos dan las brechas en pista
            gaps = np.diff(times)
            opportunities += int(np.sum(gaps <= 1.0))

        if opportunities > 0:
            empirical_prob = overtakes / opportunities
            # Escalar el valor empírico para adaptarlo al modelo del simulador
            # Un factor de escala de 2.5 y límites entre 0.02 y 0.18 es equilibrado
            base_prob = empirical_prob / 2.5
            final_prob = max(0.02, min(0.18, base_prob))
            print(f"\n[Adelantamientos] Detectados: {overtakes} overtakes, {opportunities} opportunities")
            print(f"                  Prob. Empírica: {empirical_prob:.4f} -> Prob. Base Escalada: {final_prob:.4f}")
            return round(final_prob, 4)
        
    except Exception as e:
        print(f"  [AVISO] No se pudo calcular la probabilidad empírica: {e}. Usando default.")
        
    return default_prob


def build_gp_data(year: int, gp: str) -> dict:
    """
    Extrae y procesa todos los datos necesarios para la simulación.
    """
    # ── Clasificación ──────────────────────────────────────────────
    quali = load_session(year, gp, "Q")
    qualy_results, driver_num_to_car_id = get_qualifying_results(quali)
    pole_time = qualy_results[0][1] if qualy_results else 70.0

    print(f"\n[Clasificación] Pole: {pole_time}s")
    for car_id, t, driver_name in qualy_results[:5]:
        print(f"  P{car_id+1} ({driver_name}): {t}s  (gap: +{t - pole_time:.3f}s)")

    # ── Carrera ────────────────────────────────────────────────────
    race = load_session(year, gp, "R")
    total_laps = int(race.laps["LapNumber"].max())
    circuit_name = race.event["EventName"]

    print(f"\n[Carrera] Circuito: {circuit_name}  –  {total_laps} vueltas")
    stint_data = get_stint_data(race)

    # ── Ajuste de curvas cuadráticas ───────────────────────────────
    print("\n[Ajuste cuadrático] Coeficientes por compuesto (offset respecto a pole):")
    tire_coefficients = {}
    for compound in TARGET_COMPOUNDS:
        data = stint_data[compound]
        coeffs = fit_quadratic(compound, data, pole_time)
        tire_coefficients[compound] = coeffs

    # ── Estrategias observadas ─────────
    opponent_strategies = extract_opponent_strategies(race, driver_num_to_car_id)

    # ── Resultados reales ──────────────
    real_results = extract_real_results(race, driver_num_to_car_id)

    # ── Probabilidad de adelantamiento ──
    base_overtaking_prob = calculate_overtaking_probability(race)

    return {
        "name": circuit_name,
        "year": year,
        "total_laps": total_laps,
        "pole_time": pole_time,
        "tire_coefficients": tire_coefficients,
        "qualifying_results": qualy_results,
        "opponent_strategies": opponent_strategies,
        "real_results": real_results,
        "base_overtaking_prob": base_overtaking_prob,
    }


def extract_real_results(race_session, driver_num_to_car_id: dict[str, int]) -> dict[str, dict]:
    """
    Extrae los resultados reales de carrera y las estrategias para cada piloto.
    El dict devuelto mapea car_id (como string) a un dict con:
      - driver_name
      - grid_position
      - final_position
      - status
      - total_time_seconds
      - strategy
    """
    laps = race_session.laps
    results = race_session.results

    real_results = {}

    # Encontrar tiempo del ganador de la carrera (primer piloto clasificado que no sea DNF)
    winner_row = results[results["Position"] == 1.0]
    winner_time_s = 0.0
    if not winner_row.empty:
        w_time = winner_row.iloc[0]["Time"]
        if not isinstance(w_time, float) and hasattr(w_time, "total_seconds"):
            winner_time_s = w_time.total_seconds()

    for _, row in results.iterrows():
        driver_num = str(row["DriverNumber"])
        driver_name = str(row["FullName"])
        grid_pos = int(row["GridPosition"])
        
        car_id = driver_num_to_car_id.get(driver_num)
        if car_id is None:
            # Fallback a posición en parrilla
            car_id = max(0, grid_pos - 1)

        final_pos = int(row["Position"]) if not np.isnan(row["Position"]) else 20
        status = str(row["Status"])

        # Calcular total_time_seconds
        total_time_s = None
        if status in ("Finished", "Lapped") or "Lap" in status:
            if final_pos == 1:
                total_time_s = winner_time_s
            else:
                row_time = row["Time"]
                if not isinstance(row_time, float) and hasattr(row_time, "total_seconds"):
                    total_time_s = winner_time_s + row_time.total_seconds()

        # Extraer estrategia de este piloto
        stops = []
        driver_laps = laps[laps["DriverNumber"] == driver_num].sort_values("LapNumber")
        prev_compound = None
        starting_compound = None
        for _, lap_row in driver_laps.iterrows():
            compound = str(lap_row["Compound"]).upper()
            lap_num = int(lap_row["LapNumber"])
            if compound in TARGET_COMPOUNDS:
                if starting_compound is None:
                    starting_compound = compound
                if prev_compound is not None and compound != prev_compound:
                    stops.append({"in_lap": lap_num, "tire_type": compound})
                prev_compound = compound

        real_results[str(car_id)] = {
            "driver_name": driver_name,
            "grid_position": grid_pos,
            "final_position": final_pos,
            "status": status,
            "total_time_seconds": round(total_time_s, 3) if total_time_s is not None else None,
            "starting_compound": starting_compound,
            "strategy": stops
        }

    return real_results


def extract_opponent_strategies(race_session, driver_num_to_car_id: dict[str, int]) -> dict[str, list[dict]]:
    """
    Extrae las paradas en boxes reales de cada piloto.
    car_id = posición de clasificación - 1 (0-indexed).
    """
    laps = race_session.laps
    results = race_session.results

    strategies: dict[str, list[dict]] = {}

    for driver_num, group in laps.groupby("DriverNumber"):
        driver_num_str = str(driver_num)
        car_id = driver_num_to_car_id.get(driver_num_str)
        if car_id is None:
            # Fallback a posición en parrilla
            row = results[results["DriverNumber"] == driver_num_str]
            if not row.empty:
                grid = int(row.iloc[0].get("GridPosition", 0))
                car_id = max(0, grid - 1)
            else:
                continue

        if car_id <= 0:  # Excluimos car_id=0 (usuario)
            continue

        stops = []
        group_sorted = group.sort_values("LapNumber")
        prev_compound = None
        for _, row in group_sorted.iterrows():
            compound = str(row["Compound"]).upper()
            lap_num = int(row["LapNumber"])
            if prev_compound is not None and compound != prev_compound and compound in TARGET_COMPOUNDS:
                stops.append({"in_lap": lap_num, "tire_type": compound})
            prev_compound = compound

        if stops:
            strategies[str(car_id)] = stops

    return strategies


def save_to_json(data: dict, output_path: Path):
    """Serializa el diccionario a JSON (convierte tuples a listas)."""
    def convert(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [deep_convert(i) for i in obj]
        return convert(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deep_convert(data), f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Datos guardados en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extrae datos de un GP con FastF1 y genera un JSON para la simulación.")
    parser.add_argument("--year", type=int, required=True, help="Año del campeonato (ej: 2024)")
    parser.add_argument("--gp", type=str, required=True, help="Nombre del GP (ej: 'Spanish')")
    parser.add_argument("--output", type=str, default=None, help="Ruta del archivo JSON de salida")
    args = parser.parse_args()

    # Nombre de archivo por defecto
    if args.output is None:
        slug = args.gp.lower().replace(" ", "_")
        args.output = f"data/{slug}_{args.year}.json"

    output_path = Path(args.output)
    if not output_path.is_absolute():
        if not args.output.startswith("data/") and not args.output.startswith("data\\"):
            output_path = Path(__file__).parent / "data" / args.output
        else:
            output_path = Path(__file__).parent / args.output

    # Asegurar que el directorio existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gp_data = build_gp_data(year=args.year, gp=args.gp)
        save_to_json(gp_data, output_path)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
