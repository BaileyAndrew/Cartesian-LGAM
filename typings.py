import numpy as np
from typing import Callable

type FactorMatrix = np.ndarray
type InputData = np.ndarray
type ObjectiveOracle = Callable[[list[FactorMatrix], bool], float | tuple[float, float]]
type ProjectionOracle = Callable[[list[FactorMatrix]], list[FactorMatrix]]
type ProximalOracle = Callable[[list[FactorMatrix]], float]
type GradientOracle = Callable[[list[FactorMatrix]], Direction]
type HessianUpdateOracle = Callable[[list[FactorMatrix]], Direction]
type Direction = list[FactorMatrix]
type StepSize = float
type Momentum = float
type Diagnostics = tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]
type Axis = int