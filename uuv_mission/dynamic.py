from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from uuv_mission import mission
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """
        Load mission data from CSV.

        Expected CSV layout (flexible):
        - preferred columns: reference, cave_depth, cave_height
        - fallback: if only two columns exist, treat them as (reference, cave_depth)
        The method will try data/file_name and file_name directly.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np

        p = Path(file_name)
        # try data/ fallback
        if not p.exists():
            alt = Path("data") / p
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"Mission file not found: {file_name}")

        df = pd.read_csv(p)
        # normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        # heuristics to parse common layouts
        if {"reference", "cave_depth", "cave_height"}.issubset(set(df.columns)):
            reference = pd.to_numeric(df["reference"], errors="coerce").to_numpy()
            cave_depth = pd.to_numeric(df["cave_depth"], errors="coerce").to_numpy()
            cave_height = pd.to_numeric(df["cave_height"], errors="coerce").to_numpy()
        elif {"reference", "depth"}.issubset(set(df.columns)) or df.shape[1] >= 2:
            # fallback: try (reference, cave_depth) or first two columns
            if "reference" in df.columns and "depth" in df.columns:
                reference = pd.to_numeric(df["reference"], errors="coerce").to_numpy()
                cave_depth = pd.to_numeric(df["depth"], errors="coerce").to_numpy()
            else:
                reference = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
                cave_depth = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
            # create a simple cave_height slightly above reference if not present
            span = float(np.nanmax(cave_depth) - np.nanmin(cave_depth)) if np.isfinite(np.nanmax(cave_depth)) else 1.0
            cave_height = np.full_like(reference, fill_value=np.max(reference) + abs(span)*0.5)
        else:
            raise ValueError("mission CSV must contain at least reference and cave_depth columns")

        if np.isnan(reference).any() or np.isnan(cave_depth).any():
            raise ValueError("mission.csv contains non-numeric values in required columns")

        return cls(reference=np.asarray(reference),
                   cave_height=np.asarray(cave_height),
                   cave_depth=np.asarray(cave_depth))


class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()

            # compute error between reference and measured depth
            r_t = mission.reference[t]
            error = r_t - observation_t

            # compute control action using the provided controller
            actions[t] = self.controller.compute(error)

            # apply action + disturbance to plant and step to next state
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
