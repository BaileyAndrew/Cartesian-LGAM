import warnings
import numpy as np
from typing import Optional
from typings import (
  ObjectiveOracle,
  GradientOracle,
  ProximalOracle,
  ProjectionOracle,
  Direction,
  FactorMatrix,
  StepSize,
  InputData,
  Momentum,
  Diagnostics,
  Axis
)

def proximal_backtracking_line_search(
    f: ObjectiveOracle,
    prox: ProximalOracle,
    P: ProjectionOracle,
    del_f: Direction,
    Ls: list[FactorMatrix],
    init_alpha: StepSize,
    tau: float,
    beta: float
) -> list[FactorMatrix]:
    """
    Backtracking line search
    """
    alpha = init_alpha
    
    cur_f = f(Ls)
    while True:
        new_Ls = prox(P([L - alpha * d for L, d in zip(Ls, del_f)]))
        new_f = f(new_Ls)
        prox_direc = [(L - Lnew) for L, Lnew in zip(Ls, new_Ls)]
        grad_norm = sum([np.linalg.norm(d)**2 for d in prox_direc])
        if new_f <= cur_f - beta * grad_norm / alpha:
            break
        alpha *= tau
        if alpha < 1e-30:
            warnings.warn("Alpha too small, stopping line search")
            return Ls

    return new_Ls

def proximal_accelerated_gradient(
    f: ObjectiveOracle,
    del_f: GradientOracle,
    prox: ProximalOracle,
    P: ProjectionOracle,
    Ls: list[FactorMatrix],
    mu: Momentum,
    init_alpha: StepSize,
    tau: float,
    beta: float,
    max_iter: int = 1000,
    tol: float = 1e-5,
    _momentum_import: Optional[list[FactorMatrix]] = None,
    return_momentum: bool = False,
    _delay_proximal_until: int = 1
) -> tuple[list[FactorMatrix], Diagnostics] | tuple[list[FactorMatrix], Diagnostics, list[FactorMatrix]]:
    """
    Gradient descent algorithm
    """

    cur_f = f(Ls)

    objs = []
    eps = []
    grads = []
    nonzeros = [[] for _ in Ls]
    
    for i in range(max_iter):
        grad = del_f(Ls)
        if i > 0:
            _momentum_import = ([(L - old_L) for L, old_L in zip(Ls, oldest_Ls)])
            accelerated_Ls = P([L + mu * mom for L, mom in zip(Ls, _momentum_import)])
        elif _momentum_import is not None:
            accelerated_Ls = P([L + mu * mom for L, mom in zip(Ls, _momentum_import)])
        else:
            accelerated_Ls = Ls
        oldest_Ls = Ls

        # When warm starting, it's helpful to turn off L1 for a few iterations...
        if _delay_proximal_until <= i:
            _prox = prox
        else:
            _prox = lambda x: x

        Ls = proximal_backtracking_line_search(f, _prox, P, grad, accelerated_Ls, init_alpha, tau, beta)

        new_f = f(Ls)
        delta = new_f - cur_f
        objs.append(new_f)
        eps.append(delta)
        grads.append(sum([np.linalg.norm(g) for g in grad]))

        for i, L in enumerate(Ls):
            nonzeros[i].append(np.count_nonzero(L))
        
        if np.abs(delta) < tol:
            break
        cur_f = new_f

    if return_momentum:
        return Ls, (objs, eps, grads, nonzeros), _momentum_import
    else:
        return Ls, (objs, eps, grads, nonzeros)

def warm_start(
    X: InputData,
    L_init: list[FactorMatrix],
    *,
    glassoregs: list[float] | list[tuple[float]],
    frobreg: Optional[float] = None,
    sample_axes: set[Axis] = set({}),
    mu: Momentum = 0.2,
    init_alpha: StepSize = 1,
    tau: float = 0.5,
    beta: float = 0.0001,
    max_iter: int = 50000,
    tol: float = 1e-20,
    verbose: bool = False,
    dont_warm_start: bool = False
) -> tuple[list[list[FactorMatrix]], dict[float, Diagnostics]]:
    Ls = [L.copy() for L in L_init]
    try:
        diagnostics = dict({glassoreg: None for glassoreg in glassoregs})
    except TypeError:
        raise TypeError(
            "Glassoreg should either be list[float], or list[tuple[float]], "
            + "and never list[list[float]]."
        )
    momentum = None

    outputs = []
    for glassoreg in glassoregs:
        if verbose:
            print(f"L1 Param: {glassoreg}")
        objective, gradient, proximal = get_optimizer_oracles(
            X,
            frobreg=frobreg,
            glassoregs=glassoreg,
            sample_axes=sample_axes
        )

        new_Ls, diags, momentum = proximal_accelerated_gradient(
            f=objective,
            del_f=gradient,
            prox=proximal,
            P=project_to_lower_with_positive_diag,
            Ls=Ls,
            mu=mu,
            init_alpha=init_alpha,
            tau=tau,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            _momentum_import=momentum,
            return_momentum=True,
            _delay_proximal_until=1
        )
        diagnostics[glassoreg] = diags
        outputs.append(new_Ls)

        # Perturb slightly
        Ls = [new_L + 0 * glassoreg[i] for i, new_L in enumerate(new_Ls)]
        if dont_warm_start:
            Ls = [L.copy() for L in L_init]
    return outputs, diagnostics


def get_optimizer_oracles(
    X: InputData,
    frobreg: float = None,
    glassoregs: Optional[tuple[float, ...] | float] = None,
    sample_axes: set[Axis] = set({})
) -> tuple[ObjectiveOracle, GradientOracle, ProximalOracle]:
    """
    Creates oracles from our input data

    Sample-axes are treated as known-to-be-independent, and are not updated;
        they will stay at whatever they are initialized to be (typically the identity matrix)
    """
    if frobreg is None:
        frobreg = 1/np.prod(X.shape)**(1/len(X.shape))
    if glassoregs is None:
        glassoregs = [0 for _ in range(X.ndims)]
    if isinstance(glassoregs, float) | isinstance(glassoregs, int):
        glassoregs = [glassoregs * 1.0 for _ in range(X.ndim)]

    objective: ObjectiveOracle
    def objective(Ls: list[FactorMatrix], decomp=False) -> float:
        if (len(Ls) <= 1):
            raise NotImplementedError("Must be at least matrix-variate!")

        trace_term = 0
        for i, L in enumerate(Ls):
            trace_term += np.einsum("ij,j...->i...", L, X.swapaxes(0, i)).swapaxes(0, i)
        trace_term = (trace_term**2).sum()

        log_term = np.diag(Ls[0])
        for L in Ls[1:]:
            log_term = np.add.outer(log_term, np.diag(L))
        log_term = -2 * np.log(log_term).sum()

        frob_term = 0
        for L in Ls:
            frob_term += (L**2).sum()
        frob_term *= frobreg
        
        differentiable_term = trace_term + log_term + frob_term

        if (np.isnan(differentiable_term)):
            raise ValueError("NaN in differentiable term")
        

        nondifferentiable_term = sum([glassoregs[i] * np.abs(np.tril(L)).sum() for i, L in enumerate(Ls)])
        if decomp:
            return differentiable_term, nondifferentiable_term

        return differentiable_term + nondifferentiable_term
    
    gradient: GradientOracle
    def gradient(Ls: list[FactorMatrix]) -> Direction:
        if (len(Ls) <= 1):
            raise NotImplementedError("Must be at least matrix-variate!")
        
        grads = [None] * len(Ls)

        full_log_term = np.diag(Ls[0])
        for L in Ls[1:]:
            full_log_term = np.add.outer(full_log_term, np.diag(L))
        full_log_term = (-2 / full_log_term)

        for i, L in enumerate(Ls):
            if i in sample_axes:
                grads[i] = np.zeros_like(L)
                continue
            X_i = np.swapaxes(X, i, 0).reshape(L.shape[0], -1)
            trace_term = L @ X_i @ X_i.T
            for j, other_L in enumerate(Ls):
                if i == j:
                    continue
                X_ij = np.moveaxis(X, [i, j], [0, 1]).reshape(L.shape[0], other_L.shape[0], -1)
                trace_term += np.einsum("ijc,kj,lkc->il", X_ij, other_L, X_ij)
            trace_term = np.tril(2 * trace_term)

            log_term = np.diag(full_log_term.swapaxes(i, 0).reshape(L.shape[0], -1).sum(axis=1))

            frob_term = 2*frobreg*L

            grads[i] = trace_term + log_term + frob_term

        return grads
    
    proximal: ProximalOracle
    def proximal(Ls: list[FactorMatrix]) -> list[FactorMatrix]:
        """
        Soft thresholding on each of the Ls, not affecting their diagonals
        """
        new_Ls = []
        for i, L in enumerate(Ls):
            if i in sample_axes:
                new_Ls.append(L)
                continue
            L_diag = np.diag(L)
            new_L = np.sign(L) * np.maximum(np.abs(L) - glassoregs[i], 0)
            np.fill_diagonal(new_L, L_diag)
            new_Ls.append(new_L)
        return new_Ls
    
    return objective, gradient, proximal

project_to_lower_with_positive_diag: ProjectionOracle
def project_to_lower_with_positive_diag(Ms: list[FactorMatrix], tol=1e-6) -> list[FactorMatrix]:
    """
    Project M to strictly lower-triangular + positive diagonal
    """

    Ls = []
    for M in Ms:
        L = np.tril(M)
        np.fill_diagonal(L, np.maximum(np.diag(L), tol))  # Ensure positive diag
        Ls.append(L)
    return Ls