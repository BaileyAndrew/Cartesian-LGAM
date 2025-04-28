import numpy as np
import scipy.linalg as linalg
from typing import Callable
from typings import FactorMatrix, InputData, Axis

def generate_Ls(
    dims: tuple[int, ...],
    sparsity: float,
    sample_axes: set[Axis] = set({}),
    multiply_factor: float = 0.5
) -> tuple[list[FactorMatrix], list[FactorMatrix]]:
    Ls = []
    L_patterns = []

    for i, d in enumerate(dims):
        if i in sample_axes:
            Ls.append(np.eye(d))
            L_patterns.append(np.eye(d) != 0)
            continue
        L_star = multiply_factor * np.tril(np.random.randn(d, d))

        # generate bernoulli sparsity pattern
        L_star[np.random.rand(*L_star.shape) > sparsity] = 0
        L_pattern = (np.abs(np.tril(L_star, k=-1)) > 1e-16)
        np.fill_diagonal(L_star, 1)
        np.fill_diagonal(L_pattern, 0)

        Ls.append(L_star)
        L_patterns.append(L_pattern)

    return Ls, L_patterns

def generate_sylvester_data(
        Ls: list[FactorMatrix],
        normalize: bool=False,
        source_distribution: Callable[[tuple[int,...]], np.ndarray] = np.random.normal
    ) -> InputData:
    """
    Generates data that follows the tensor-variate sylvester equation
    """
    n = [L.shape[0] for L in Ls]
    dim = len(n)

    if dim == 2:
        # Matrix variate equation sub-case; already a SciPy routine to do this!
        # Note that the routine has some preprocessing steps to triangularize our
        # matrices, which is not needed here, so to squeeze out extra efficiency
        # one could drop down to the lapack dtrsyl routine.
        X = linalg.solve_sylvester(Ls[0], Ls[1], source_distribution(size=n))
    else:
        # There is literature on solving this, but in general the problem is
        # more difficult and there do not exist SciPy routines.
        # So, we'll take the 'lazy way' and reduce it to the dim=2 case.
        # Note that this has egregious space requirements.  In general,
        # if all K axes have size d, then this algorithm transforms the
        # data into two d^(K // 2) x d^(K // 2) matrices.
        # That's a bit expensive...
        #
        # Essentially, we can handle small tensors up to dim=4, after which
        # all bets are off!
        left_Ls = Ls[:(dim // 2)]
        right_Ls = Ls[(dim // 2):]

        left_L = left_Ls[0]
        for L in left_Ls[1:]:
            left_L = np.kron(left_L, np.eye(L.shape[0])) + np.kron(np.eye(left_L.shape[0]), L)
        right_L = right_Ls[0]
        for L in right_Ls[1:]:
            right_L = np.kron(right_L, np.eye(L.shape[0])) + np.kron(np.eye(right_L.shape[0]), L)
        
        X = linalg.solve_sylvester(
            left_L,
            right_L,
            source_distribution(size=(left_L.shape[0], right_L.shape[0]))
        ).reshape(n)

    if normalize:
        X /= np.sqrt((X**2).sum())
    return X