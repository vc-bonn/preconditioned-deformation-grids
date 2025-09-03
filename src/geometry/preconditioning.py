import torch
# from largesteps.geometry import laplacian_uniform
from largesteps.parameterize import from_differential

def laplacian_uniform(verts, faces):
    """
    Taken from the original largesteps implementation, fixing the cuda device
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

class Preconditioner(torch.nn.Module):
    def __init__(
        self, opt_args: dict, vertex_features: torch.Tensor, faces: torch.Tensor
    ):
        super().__init__()
        self.opt_args = opt_args

        self.v = vertex_features.to(torch.float32)
        self.V = self.v.shape[0]

        if faces is not None:
            self.f = faces.to(torch.int32)
            self.F = self.f.shape[0]
        else:
            self.f = faces

    def _precondition(self):
        self.u = self.M @ self.v
        self.u.requires_grad_()

    def _compute_matrix(self):
        self.L = laplacian_uniform(self.v, self.f)
        idx = torch.arange(self.V, dtype=torch.long, device=self.v.device)
        self.eye = torch.sparse_coo_tensor(
            torch.stack((idx, idx), dim=0),
            torch.ones(self.V, dtype=torch.float, device=self.v.device),
            (self.V, self.V),
        )
        self.M = torch.add(self.eye, self.smoothness * self.L).coalesce()

    @property
    def smoothness(self) -> int:
        return self.opt_args["smoothness"]

    @property
    def solver(self) -> str:
        return self.opt_args["solver"]

    @property
    def lr(self) -> str:
        return self.opt_args["lr"]

    def forward(self):
        # return (from_differential(self.M, self.u, method=self.solver), self.f)
        return (from_differential(self.M, self.u, method="CG"), self.f)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
