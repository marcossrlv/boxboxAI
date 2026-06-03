"""
simulation_logger.py
────────────────────
Registra el estado de cada vuelta durante una simulación de carrera y genera:

  - logs/<run_id>/laps.csv      → una fila por coche por vuelta
  - logs/<run_id>/summary.json  → resumen del episodio completo

Uso básico:
    logger = SimulationLogger(gp_name="Spanish GP 2024")
    # dentro del loop de episodio:
    logger.log_lap(lap=1, race_state=state, agent_action=0)
    # al final del episodio:
    logger.close(final_state=state)

Uso con RaceGymEnv (se pasa como parámetro):
    env = RaceGymEnv(gp_data=gp, logger=SimulationLogger("Spanish GP 2024"))
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# Directorio raíz donde se guardan los logs
LOGS_ROOT = Path(__file__).parent / "logs"

# Columnas del CSV de vueltas
_LAP_CSV_FIELDS = [
    "run_id",
    "episode",
    "lap",
    "car_id",
    "is_agent",
    "agent_action",       # solo relevante cuando is_agent=True
    "lap_time",
    "total_time",
    "position",           # posición en pista en esa vuelta (1-indexed)
    "tire_type",
    "laps_on_tire",
    "tire_wear",          # laps_on_tire / track_laps
    "fuel_mass",
    "pit_stop",           # True si hubo parada este lap
    "is_finished",
]

_ACTION_NAMES = {
    None: "none",
    0: "no_pit",
    1: "pit_soft",
    2: "pit_medium",
    3: "pit_hard",
}


class SimulationLogger:
    """
    Logger de simulaciones de carrera.

    Args:
        gp_name:    Nombre del GP (usado en el resumen y nombre del directorio).
        log_dir:    Directorio base donde guardar logs. Por defecto: ./logs/
        run_id:     Identificador único del run. Si None, se genera con timestamp.
    """

    def __init__(
        self,
        gp_name: str = "unknown",
        log_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        self.gp_name = gp_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode = 0
        self.track_laps: Optional[int] = None  # se setea en el primer log_lap

        # Crear directorio del run
        root = Path(log_dir) if log_dir else LOGS_ROOT
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Abrir CSV de vueltas (modo append para acumular episodios)
        self._csv_path = self.run_dir / "laps.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=_LAP_CSV_FIELDS)
        self._writer.writeheader()

        # Buffer para el resumen del episodio actual
        self._episode_laps: list[dict] = []
        self._episode_pit_stops: list[dict] = []
        self._episode_start_time = datetime.now()

        # Resumen acumulado de todos los episodios
        self._all_episodes: list[dict] = []

        print(f"[Logger] Run '{self.run_id}' -> {self.run_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def on_reset(self, track_laps: int):
        """Llamar al inicio de cada episodio (env.reset)."""
        self.episode += 1
        self.track_laps = track_laps
        self._episode_laps = []
        self._episode_pit_stops = []
        self._episode_start_time = datetime.now()

    def log_lap(
        self,
        race_state: dict,
        agent_car_id: int = 0,
        agent_action: Optional[int] = None,
        track_laps: Optional[int] = None,
    ):
        """
        Registra el estado de una vuelta completa.

        Args:
            race_state:    Dict devuelto por Race.get_race_state() / step().
            agent_car_id:  car_id del agente (para marcar is_agent y acción).
            agent_action:  Acción tomada por el agente en esta vuelta.
            track_laps:    Total de vueltas del circuito (para calcular tire_wear).
        """
        if track_laps is not None:
            self.track_laps = track_laps

        lap = race_state.get("time_step", 0)
        order = race_state.get("order", [])
        prev_tires: dict = getattr(self, "_prev_tires", {})

        for pos, car_id in enumerate(order, start=1):
            tire_type = race_state.get("tire_type", {}).get(car_id, "")
            laps_on = race_state.get("laps_on_tire", {}).get(car_id, 0)
            fuel = race_state.get("fuel_mass", {}).get(car_id, 0.0)
            lap_time = race_state.get("lap_times", {}).get(car_id, 0.0)
            total_time = race_state.get("total_time", {}).get(car_id, 0.0)
            finished = race_state.get("finished", {}).get(car_id, False)

            # Detectar pit stop: laps_on_tire bajó respecto a la vuelta anterior
            prev_laps_on = prev_tires.get(car_id, {}).get("laps_on", -1)
            pit_stop = (prev_laps_on > 0 and laps_on < prev_laps_on)

            if pit_stop and car_id != agent_car_id:
                self._episode_pit_stops.append({
                    "lap": lap,
                    "car_id": car_id,
                    "new_tire": tire_type,
                })

            tire_wear = (
                min(1.0, laps_on / self.track_laps) if self.track_laps else 0.0
            )

            row = {
                "run_id":       self.run_id,
                "episode":      self.episode,
                "lap":          lap,
                "car_id":       car_id,
                "is_agent":     car_id == agent_car_id,
                "agent_action": _ACTION_NAMES.get(agent_action, str(agent_action))
                                if car_id == agent_car_id else "",
                "lap_time":     round(lap_time, 3),
                "total_time":   round(total_time, 3),
                "position":     pos,
                "tire_type":    tire_type,
                "laps_on_tire": laps_on,
                "tire_wear":    round(tire_wear, 4),
                "fuel_mass":    round(float(fuel), 2),
                "pit_stop":     pit_stop,
                "is_finished":  finished,
            }
            self._writer.writerow(row)
            self._episode_laps.append(row)

            # Actualizar estado previo para detección de pit stops
            if car_id not in prev_tires:
                prev_tires[car_id] = {}
            prev_tires[car_id]["laps_on"] = laps_on

        self._prev_tires = prev_tires

    def close_episode(self, final_state: Optional[dict] = None):
        """
        Finaliza el episodio actual: escribe el resumen JSON.
        Llamar cuando el episodio termina (terminated=True).
        """
        duration = (datetime.now() - self._episode_start_time).total_seconds()
        winner = final_state.get("winner") if final_state else None

        # Clasificación final por tiempo total
        leaderboard = final_state.get("leaderboard_by_time", []) if final_state else []

        # Estadísticas del agente
        agent_rows = [r for r in self._episode_laps if r["is_agent"]]
        agent_pit_count = sum(1 for r in agent_rows if r["pit_stop"])
        agent_best_lap = min((r["lap_time"] for r in agent_rows), default=0.0)
        agent_final_pos = next(
            (r["position"] for r in reversed(agent_rows)), None
        )

        summary = {
            "run_id":           self.run_id,
            "gp_name":          self.gp_name,
            "episode":          self.episode,
            "total_laps":       self.track_laps,
            "duration_seconds": round(duration, 2),
            "winner_car_id":    winner,
            "leaderboard":      [[car_id, round(t, 3)] for car_id, t in leaderboard],
            "agent": {
                "final_position": agent_final_pos,
                "pit_stops":      agent_pit_count,
                "best_lap_time":  round(agent_best_lap, 3),
                "total_laps_run": len(agent_rows),
            },
            "opponent_pit_stops": self._episode_pit_stops,
        }

        self._all_episodes.append(summary)
        self._flush_summary()
        return summary

    def close(self):
        """Cierra el CSV. Llamar al destruir el env."""
        self._csv_file.flush()
        self._csv_file.close()
        print(f"[Logger] Logs guardados en: {self.run_dir}")
        print(f"  laps.csv      -> {self._csv_path}")
        print(f"  summary.json  -> {self.run_dir / 'summary.json'}")

    # ──────────────────────────────────────────────────────────────────────────
    # Internos
    # ──────────────────────────────────────────────────────────────────────────

    def _flush_summary(self):
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._all_episodes, f, indent=2, ensure_ascii=False)

    def __del__(self):
        try:
            if not self._csv_file.closed:
                self._csv_file.close()
        except Exception:
            pass
