from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class Mission:
    time: np.ndarray
    depth: np.ndarray
    reference: np.ndarray

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "Mission":
        p = Path(csv_path)
        if not p.exists():
            alt = Path("data") / p
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(f"Mission file not found: {csv_path}")

        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]

        if "time" in df.columns and "depth" in df.columns:
            time = pd.to_numeric(df["time"], errors="coerce").to_numpy()
            depth = pd.to_numeric(df["depth"], errors="coerce").to_numpy()
            reference = (
                pd.to_numeric(df["reference"], errors="coerce").to_numpy()
                if "reference" in df.columns
                else depth.copy()
            )
        elif df.shape[1] >= 2:
            time = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            depth = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
            reference = depth.copy()
        else:
            raise ValueError("mission.csv must contain at least time and depth columns")

        if np.isnan(time).any() or np.isnan(depth).any():
            raise ValueError("mission.csv contains non-numeric values")

        return cls(time, depth, reference)
    