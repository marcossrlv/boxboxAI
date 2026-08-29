"""
Domain Model for Race Simulation
Contains classes necessary to simulate a racing environment
"""

from typing import List, Optional
from enum import Enum
import random


class TireType(Enum):
    """Enumeration of tire types"""
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


class QuadraticTireModel:
    """Modelo de neumáticos con curva cuadrática por compuesto.
    
    Cada compuesto tiene coeficientes (a, b, c) tales que:
        offset(n) = a·n² + b·n + c
    donde n = vueltas con ese juego de neumáticos.
    
    - c: offset inicial del compuesto (ventaja/desventaja inherente)
    - b: degradación lineal por vuelta
    - a: aceleración de la degradación (efecto "cliff")
    """

    def __init__(self, coefficients: Optional[dict] = None):
        """
        Args:
            coefficients: Dict mapping TireType -> (a, b, c) tuple.
                If None, default coefficients are used.
        """
        if coefficients is None:
            self._coefficients = {
                TireType.SOFT:   (0.005, 0.03, 0.0),
                TireType.MEDIUM: (0.002, 0.015, 0.6),
                TireType.HARD:   (0.001, 0.008, 1.2),
            }
        else:
            self._coefficients = coefficients

    def get_time_offset(self, tire_type: TireType, laps_on_tire: int) -> float:
        """Returns time offset in seconds for the given compound and tire age."""
        a, b, c = self._coefficients.get(tire_type, (0.002, 0.015, 0.5))
        n = float(laps_on_tire)
        return a * n * n + b * n + c


class FuelModel:
    """Model for fuel mass simulation and its lap time impact."""

    def __init__(self, loss_per_lap: float = 1.6, penalty_k: float = 0.025):
        self.loss_per_lap = loss_per_lap
        self.penalty_k = penalty_k

    def initial_fuel_mass(self, track: "Track") -> float:
        required = self.loss_per_lap * track.laps
        margin = required * 0.05  # 5% extra fuel margin
        return float(required + margin)

    def fuel_mass_loss_per_lap(self) -> float:
        return float(self.loss_per_lap)

    def update_fuel_after_lap(self, fuel_mass: float, fuel_loss: float) -> float:
        return max(0.0, float(fuel_mass) - float(fuel_loss))

    def time_penalty_multiplier(self, fuel_mass: float) -> float:
        return 1.0 + (self.penalty_k * (float(fuel_mass) / 100.0))


class PitStopModel:
    """Model for timing and executing pit stops."""

    def __init__(self, pit_loss_seconds: float = 23.0):
        self.pit_loss_seconds = pit_loss_seconds

    def pit_time_loss_seconds(self, car: "Car", race: "Race") -> float:
        return float(self.pit_loss_seconds)

    def apply_pit(self, car: "Car", new_tire_type: Optional[TireType]) -> None:
        car.laps_on_tire = 0
        if new_tire_type is not None:
            car.current_tire_type = new_tire_type
            car.tire_history.append(new_tire_type)


class OvertakingModel:
    """Model for probabilistic overtaking between cars."""

    def __init__(self, base_prob: float = 0.10, max_gap: float = 1.0):
        self.base_prob = base_prob
        self.max_gap = max_gap

    def apply(self, cars_in_order: List["Car"], rng: random.Random, track: "Track") -> List["Car"]:

        new_order = list(cars_in_order)
        for i in range(len(new_order) - 1, 0, -1):
            behind = new_order[i]
            ahead = new_order[i - 1]

            if behind.is_finished or ahead.is_finished:
                continue

            # Comprobar el gap real (diferencia en tiempo total)
            gap = max(0.0, behind.total_time - ahead.total_time)
            if gap > self.max_gap:
                continue

            # Diferencia de ritmo en la última vuelta
            time_diff = (ahead.current_lap_time - behind.current_lap_time)
            speed_edge = max(-1.0, min(1.0, time_diff / 10.0))

            # Ventaja por neumáticos más frescos (normalizado a ~30 vueltas de diferencia)
            tire_age_diff = (ahead.laps_on_tire - behind.laps_on_tire) / 30.0
            wear_edge = max(-1.0, min(1.0, tire_age_diff))

            prob = track.base_overtaking_prob
            prob += 0.15 * max(0.0, speed_edge)
            prob += 0.05 * max(0.0, wear_edge)
            prob = max(0.0, min(0.9, prob))

            if rng.random() < prob:
                # Intercambiar posiciones en la lista de orden en pista
                new_order[i - 1], new_order[i] = new_order[i], new_order[i - 1]
                
                # Ajustar tiempos acumulados (total_time) para mantener la consistencia física
                t_anterior_delante = ahead.total_time
                behind.total_time = t_anterior_delante
                ahead.total_time = t_anterior_delante + 0.1

        return new_order


class Car:
    """Represents a racing car with various attributes"""
    
    def __init__(self, car_id: int, initial_tire_type: TireType, base_lap_time: float = 90.0, driver_name: str = None):
        self.car_id = car_id
        self.driver_name = driver_name or f"Driver {car_id}"
        self.current_tire_type = initial_tire_type
        self.base_lap_time = base_lap_time
        
        self.laps_on_tire = 0
        self.fuel_mass = 0.0
        self.total_time = 0.0
        self.strategy = {
            'planned_stops': []
        }
        self.tire_history = [initial_tire_type]
        
        # Dynamic attributes
        self.current_lap_time = 0.0
        self.lap = 1
        self.is_finished = False
        
    def complete_lap(self, tire_model: QuadraticTireModel, wear_multiplier: float = 1.0) -> float:
        """
        Complete one lap and return the time taken.
        Formula: BaseTime + TireOffset(n) + RNG
        """
        # Variación aleatoria aditiva (±0.3s)
        time_variation = random.uniform(-0.3, 0.3)
        
        # Offset de neumáticos basado en la curva cuadrática del compuesto
        tire_offset = tire_model.get_time_offset(self.current_tire_type, self.laps_on_tire)
        
        self.current_lap_time = self.base_lap_time + tire_offset + time_variation
        
        # Incrementar vueltas con este juego (afectado por aire sucio)
        self.laps_on_tire += wear_multiplier
        
        # Advance one lap
        self.lap += 1
        
        # Check if finished
        if self.lap > self.total_laps:
            self.is_finished = True
        
        return self.current_lap_time
    
    def reset(self):
        """Reset car to initial state"""
        self.current_lap_time = 0.0
        self.total_time = 0.0
        self.fuel_mass = 0.0
        self.laps_on_tire = 0
        self.lap = 1
        self.is_finished = False
        self.tire_history = [self.tire_history[0]]
        self.current_tire_type = self.tire_history[0]


class Track:
    """Represents a racing track.
    
    The base overtaking probability is typically calculated dynamically from real
    race data using `fetch_race_data.py` (based on actual green-flag overtakes
    divided by DRS/dirty air opportunities). If not provided, it falls back to
    preconfigured default values or a general baseline.
    """
    
    # Default overtaking probabilities mapped by track/event name as fallbacks
    OVERTAKING_PROBABILITIES = {
        "Spanish Grand Prix": 0.09,
    }
    
    def __init__(
        self,
        laps: int,
        base_lap_time: float,
        name: str = "",
        base_overtaking_prob: Optional[float] = None
    ):
        self.laps = laps
        self.base_lap_time = base_lap_time  # Base lap time in seconds
        self.name = name
        
        if base_overtaking_prob is not None:
            self.base_overtaking_prob = base_overtaking_prob
        else:
            # Fallback configured in the domain model
            self.base_overtaking_prob = self.OVERTAKING_PROBABILITIES.get(name, 0.10)


class Race:
    """Main race simulation class"""
    
    def __init__(
        self,
        track: Track,
        cars: List[Car],
        tire_model: Optional[QuadraticTireModel] = None,
        overtaking_model: Optional[OvertakingModel] = None,
        fuel_model: Optional[FuelModel] = None,
        pit_model: Optional[PitStopModel] = None,
        rng: Optional[random.Random] = None,
        agent_car_id: int = 1,
    ):
        self.track = track
        self.cars = cars
        self.time_step = 0
        self.max_time_steps = track.laps  # Each step is one lap
        self.race_finished = False
        self.winner = None
        self.total_race_time = 0.0
        self.tire_model = tire_model or QuadraticTireModel()
        self.overtaking_model = overtaking_model or OvertakingModel()
        self.fuel_model = fuel_model or FuelModel()
        self.pit_model = pit_model or PitStopModel()
        self.rng = rng or random.Random()
        self.agent_car_id = agent_car_id
        
        # Race state
        self.order = [car.car_id for car in cars]
        
        # Set total laps for each car
        for idx, car in enumerate(cars):
            car.starting_position = idx + 1
            car.total_laps = track.laps
            car.fuel_mass = self.fuel_model.initial_fuel_mass(track)
        
    def step(self, agent_action: Optional[int] = None) -> dict:
        """Execute one time step (one lap) with optional agent pit action."""
        self.time_step += 1

        # Update each car (complete one lap)
        car_map = {c.car_id: c for c in self.cars}
        for car in self.cars:
            if not car.is_finished:
                pit_loss = 0.0
                pit_tire_type: Optional[TireType] = None
                
                # Si es el coche del agente realizar acción del agente, si no realizar paradas planificdas
                if car.car_id == self.agent_car_id:
                    if agent_action in (1, 2, 3):
                        pit_tire_type = {
                            1: TireType.SOFT,
                            2: TireType.MEDIUM,
                            3: TireType.HARD,
                        }[agent_action]
                else:
                    planned_stops = car.strategy.get('planned_stops', [])
                    for stop in planned_stops:
                        if stop.get('in_lap') == car.lap:
                            pit_tire_type = stop.get('tire_type')
                            break

                if pit_tire_type is not None:
                    pit_loss = self.pit_model.pit_time_loss_seconds(car, self)
                    self.pit_model.apply_pit(car, pit_tire_type)

                # Calcular presencia de aire sucio (gap con coche de delante <= 1.0s en la vuelta anterior)
                dirty_air_wear_penalty = 1.0
                dirty_air_time_penalty = 0.0
                
                try:
                    current_pos = self.order.index(car.car_id)
                except ValueError:
                    current_pos = -1
                
                if current_pos > 0:
                    ahead_car_id = self.order[current_pos - 1]
                    ahead_car = car_map[ahead_car_id]
                    gap = car.total_time - ahead_car.total_time
                    if 0.0 <= gap <= 1.0:
                        dirty_air_wear_penalty = 1.1
                        dirty_air_time_penalty = 0.2

                lap_time = car.complete_lap(self.tire_model, wear_multiplier=dirty_air_wear_penalty)

                lap_time *= self.fuel_model.time_penalty_multiplier(car.fuel_mass)
                fuel_loss = self.fuel_model.fuel_mass_loss_per_lap()
                car.fuel_mass = self.fuel_model.update_fuel_after_lap(car.fuel_mass, fuel_loss)

                lap_time += pit_loss + dirty_air_time_penalty
                car.current_lap_time = lap_time
                car.total_time += lap_time
                
        # Mapear self.order (orden de pista persistente de la vuelta anterior) a objetos Car
        car_map = {car.car_id: car for car in self.cars}
        cars_in_track_order = [car_map[car_id] for car_id in self.order]
        
        # Aplicar el modelo de adelantamiento sobre el orden real de pista
        cars_in_track_order = self.overtaking_model.apply(cars_in_track_order, self.rng, self.track)
        self.order = [c.car_id for c in cars_in_track_order]
        
        # Check if race is finished (all cars completed all laps)
        if all(car.is_finished for car in self.cars) or self.time_step >= self.max_time_steps:
            self.race_finished = True
            # Determine winner based on the minimum total_time of all cars
            self.winner = min(self.cars, key=lambda c: c.total_time)
        
        return self.get_race_state()
    
    def get_race_state(self) -> dict:
        """Get current race state"""
        sorted_by_time = sorted(self.cars, key=lambda c: c.total_time)
        
        return {
            'time_step': self.time_step,
            'driver_names': {car.car_id: car.driver_name for car in self.cars},
            'lap_times': {car.car_id: car.current_lap_time for car in self.cars},
            'laps': {car.car_id: car.lap for car in self.cars},
            'laps_on_tire': {car.car_id: car.laps_on_tire for car in self.cars},
            'fuel_mass': {car.car_id: car.fuel_mass for car in self.cars},
            'total_time': {car.car_id: car.total_time for car in self.cars},
            'order': list(self.order),
            'finished': {car.car_id: car.is_finished for car in self.cars},
            'leaderboard_by_time': [(car.car_id, car.total_time) for car in sorted_by_time],
            'race_finished': self.race_finished,
            'winner': self.winner.car_id if self.winner else None
        }
    
    def reset(self):
        """Reset race to initial state"""
        self.time_step = 0
        self.race_finished = False
        self.winner = None
        self.order = [car.car_id for car in self.cars]
        
        for car in self.cars:
            car.reset()
            car.fuel_mass = self.fuel_model.initial_fuel_mass(self.track)



