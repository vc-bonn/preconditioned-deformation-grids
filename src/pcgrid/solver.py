import torch
from largesteps.solvers import CholeskySolver, ConjugateGradientSolver, solve
import weakref

"""
Originally from:
Large Steps in Inverse Rendering of Geometry [https://github.com/rgl-epfl/large-steps-pytorch/tree/master]

Extends the ConjugateGradientSolver to handle batched solving. 
Improves the performance by allowing the solver to handle multiple right-hand sides at once.
Especially useful for optimization tasks where a lot of values/dimensions are optimized simultaneously.
"""


class Batched_CG_Solver(ConjugateGradientSolver):
    def __init__(self, M, T_max, T=0):
        super().__init__(M)
        self.T = T
        self.T_max = T_max

    def update_T(self, T):
        """
        Update the current timestep T.
        """
        self.T = T

    def solve(self, b, backward=False):
        """
        Solve the sparse linear system.

        There is actually one linear system to solve for each axis in b
        (typically x, y and z), and we have to solve each separately with CG.
        Therefore this method calls self.solve_axis for each individual system
        to form the solution.

        Parameters
        ----------
        b : torch.Tensor
            The right hand side of the system Ax=b.
        backward : bool
            Whether we are in the backward or the forward pass.
        """
        if self.guess_fwd is None:
            # Initialize starting guesses in the first run
            self.guess_bwd = torch.zeros_like(b)
            self.guess_fwd = torch.zeros_like(b)

        if backward:
            x0 = self.guess_bwd
        else:
            x0 = self.guess_fwd

        if len(b.shape) != 2:
            raise ValueError(
                f"Invalid array shape {b.shape} for ConjugateGradientSolver.solve: expected shape (a, b)"
            )

        x = self.solve_batch(b, x0)

        if backward:
            self.guess_bwd = x
        else:
            self.guess_fwd = x

        return x

    def solve_batch(self, b, x0):
        x = torch.zeros_like(b) if x0 is None else x0
        r = b - self.M @ x
        p = r.clone()
        rs_old = torch.sum(r * r, dim=0)
        for _ in range(100):  # r_norm > 1e-5:
            Ap = self.M @ p
            alpha = rs_old / (torch.sum(p * Ap, dim=0) + 1e-8)
            x = x + alpha.unsqueeze(0) * p
            r = r - alpha.unsqueeze(0) * Ap
            rs_new = torch.sum(r * r, dim=0)
            if torch.all(rs_new < 1e-4):
                break
            if (
                torch.isnan(alpha).any()
                or torch.isinf(alpha).any()
                or torch.isnan(p).any()
                or torch.isinf(p).any()
            ):
                return x

            beta = rs_new / (rs_old + 1e-8)
            p = r + beta.unsqueeze(0) * p
            rs_old = rs_new
        return x


# Cache for the system solvers
_cache = {}


def cache_put(key, value, A):
    # Called when 'A' is garbage collected
    def cleanup_callback(wr):
        del _cache[key]

    wr = weakref.ref(A, cleanup_callback)

    _cache[key] = (value, wr)


def from_differential(L, u, T_max, method="Cholesky"):
    """
    Convert differential coordinates back to Cartesian.

    If this is the first time we call this function on a given matrix L, the
    solver is cached. It will be destroyed once the matrix is garbage collected.

    Parameters
    ----------
    L : torch.sparse.Tensor
        (I + l*L) matrix
    u : torch.Tensor
        Differential coordinates
    method : {'Cholesky', 'CG'}
        Solver to use.
    """
    key = (id(L), method)
    if key not in _cache.keys():
        if method == "Cholesky":
            solver = CholeskySolver(L)
        elif method == "CG":
            solver = Batched_CG_Solver(L, T_max)
        else:
            raise ValueError(f"Unknown solver type '{method}'.")

        cache_put(key, solver, L)
    else:
        solver = _cache[key][0]

    return solve(solver, u)
