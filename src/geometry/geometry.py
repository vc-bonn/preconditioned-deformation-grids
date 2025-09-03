import torch
from largesteps.parameterize import from_differential
from src.geometry.preconditioning import Preconditioner


class Geometry(Preconditioner):
    def __init__(
        self, opt_args: dict, vertex_features: torch.Tensor, faces: torch.Tensor
    ) -> None:
        super().__init__(opt_args, vertex_features, faces)
        if faces is None:
            self.v.requires_grad_()
            self.optimizer = torch.optim.Adam([self.v], self.lr)
        else:
            self.args = opt_args

            self._compute_matrix()
            self._precondition()
            self.optimizer = torch.optim.Adam([self.u], self.lr)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def forward(self):
        return from_differential(self.M, self.u, method=self.solver), self.f
