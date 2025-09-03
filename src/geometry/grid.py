import torch
from largesteps.parameterize import to_differential, from_differential
import time
from pytorch3d.ops import knn


class Grid(torch.nn.Module):
    def __init__(self, args: dict, target_points: torch.Tensor) -> None:
        super().__init__()

        self.args = args
        goal_idx = torch.cat(
            (
                torch.arange(0, self.args.method_args["keyframe_index"]),
                torch.arange(
                    self.args.method_args["keyframe_index"] + 1, target_points.shape[0]
                ),
            )
        )
        self.points = target_points.squeeze()[goal_idx, ..., :3]
        self.T = self.points.shape[0]

        self.smoothness = [
            self.transform_args["smoothness"]
            * self.grid_args["local_lambda_adapt"] ** i
            for i in range(self.grid_args["n_level"])
        ]
        resolutions = [
            self.grid_args["base_res"] + self.grid_args["increase_per_level"] * i
            for i in range(self.grid_args["n_level"])
        ]
        self.lrs = [
            self.transform_args["lr"] * self.grid_args["lr_adapt"] ** i
            for i in range(self.grid_args["n_level"])
        ]

        print("Grid Learning Rates Level: {}".format(self.lrs))
        print("Grid Smoothness Level: {}".format(self.smoothness))
        print(
            "Points [{} x {} x {}]".format(
                self.points.shape[0], self.points.shape[1], self.points.shape[2]
            )
        )
        self.grids = []
        for res, lr, sm in zip(resolutions, self.lrs, self.smoothness):
            print("Grid [{} x {} x {} x {}]]".format(self.T, res, res, res))
            t0 = time.time()
            self.grids.append(
                GridLevel(
                    lr,
                    sm,
                    res,
                    target_points.squeeze()[..., :3],
                    min(self.grid_args["neighbours"], res - 1),
                    self.transform_args["solver"],
                )
            )
            print("   Initialization-Time [{}s]".format(round(time.time() - t0, 2)))
        self.condition(self.points)
        self.zero_grad()

    @property
    def grid_args(self):
        return self.args.method_args["grid"]

    @property
    def transform_args(self):
        return self.args.method_args["transform"]

    def condition(self, points):
        for grid in self.grids:
            grid.condition(points.squeeze()[..., :3])
            grid.grid_mask = grid.grid_mask.to(self.args.device)
            grid.opt_values = grid.opt_values.to(self.args.device)

    def step(self) -> None:
        for grid in self.grids:
            grid.optimizer.step()

    def zero_grad(self):
        for grid in self.grids:
            grid.optimizer.zero_grad()
            grid.grid = torch.zeros(
                self.T * grid.resolution * grid.resolution * grid.resolution,
                6,
                device=self.args.device,
            )
            grid.solve()

    def get(self):
        values = []
        for grid in self.grids:
            values.append(grid.grid.detach())
        return values

    def forward(self, data: dict, weight=None):
        if len(data) > 0:
            output = torch.stack([grid.forward(data) for grid in self.grids])
            if weight is None:
                return output.mean(dim=0)
            else:
                out = (output * weight.permute(2, 0, 1)[:, :, None, :]).sum(dim=0)
                return out
        else:
            return 0

    def forward_(self, data: dict, weight=None):
        if len(data) > 0:
            output = torch.stack([grid(data) for grid in self.grids])
            if weight is None:
                return output.mean(dim=0)
            else:
                out = (output * weight).sum(dim=-1)
                return out
        else:
            return 0

    def save(self):
        import os

        os.makedirs(os.path.join(self.args.io_args["out_path"], "grids"))
        for grid in self.grids:
            grid.solve()
            os.makedirs(
                os.path.join(
                    self.args.io_args["out_path"], "grids", str(grid.resolution)
                )
            )
            grid_ = grid.grid_.detach().cpu()
            mask_ = grid.grid_mask.detach().cpu()
            torch.save(
                grid_,
                os.path.join(
                    self.args.io_args["out_path"],
                    "grids",
                    str(grid.resolution),
                    "grid.pt",
                ),
            )
            torch.save(
                mask_,
                os.path.join(
                    self.args.io_args["out_path"],
                    "grids",
                    str(grid.resolution),
                    "mask.pt",
                ),
            )
        return


class GridLevel(torch.nn.Module):
    def __init__(
        self,
        lr: float,
        smoothness: float ,
        grid_res: int,
        target_points,  # T x P x 3,
        shift: int,
        solver: str,
    ):
        super().__init__()
        self.T = target_points.shape[0] - 1
        self.indices = torch.arange(
            self.T * grid_res * grid_res * grid_res,
            dtype=torch.long,
            device=target_points.device,
        )
        self.resolution = grid_res

        self.grid = torch.zeros(self.T * grid_res * grid_res * grid_res, 6)

        self.grid_mask = torch.ones_like(self.grid[:, 0], dtype=torch.bool)
        self.lr = lr
        self.smoothness = smoothness
        self.shift = shift
        self.solver = solver

    def condition(self, input_points):
        if self.resolution > 1:
            self.prune_grid(input_points, self.shift)
            self._compute_neighbours()
            self.opt_values = torch.zeros(
                (self.adj.max() + 1, 6), requires_grad=False, device=input_points.device
            )
            self._compute_matrix(self.smoothness)
            self.opt_values = to_differential(self.M, self.opt_values)
            self.opt_values.requires_grad_()
        else:
            self.opt_values = torch.zeros_like(
                self.grid[self.grid_mask], requires_grad=True
            )
        self.optimizer = torch.optim.Adam(params=[self.opt_values], lr=self.lr)
        if self.resolution > 1:
            del self.adj
            torch.cuda.empty_cache()

    def forward(self, data: dict) -> torch.Tensor:
        if data["index"].shape[0] == 1:
            return torch.nn.functional.grid_sample(
                self.grid_[data["target_index"] : data["target_index"] + 1],
                data["points"][..., :3],
                padding_mode="border",
                align_corners=True,
            ).squeeze(dim=(-3, -2))
        elif data["index"].shape[0] == self.grid_.shape[0]:
            return torch.nn.functional.grid_sample(
                self.grid_,
                data["points"][..., :3],
                padding_mode="border",
                align_corners=True,
            ).squeeze(dim=(-3, -2))
        else:
            return torch.nn.functional.grid_sample(
                self.grid_[data["target_index"]],
                data["points"][..., :3],
                padding_mode="border",
                align_corners=True,
            ).squeeze(dim=(-3, -2))


    def solve(self):
        if self.resolution > 1:
            opt_values = from_differential(self.M, self.opt_values, method=self.solver)
        else:
            opt_values = self.opt_values
        rot_values = opt_values[:, :3]
        tran_values = torch.tanh(opt_values[:, -3:]) * 0.5
        self.grid[self.grid_mask] = torch.cat((rot_values, tran_values), dim=-1)
        self.grid_ = self.grid.reshape(
            self.T, self.resolution, self.resolution, self.resolution, -1
        ).permute(0, 4, 1, 2, 3)

    def prune_grid(self, input_points: torch.Tensor, shift: int):
        device = input_points.device
        input_points = input_points.to(device)
        voxel_size = 2 / self.resolution  # grid is defined in space [-1,1]
        positions = torch.cat(
            [
                input_points + (shift_ * voxel_size)
                for shift_ in torch.flatten(
                    torch.stack(
                        torch.meshgrid(
                            torch.arange(-shift, shift + 1, device=device),
                            torch.arange(-shift, shift + 1, device=device),
                            torch.arange(-shift, shift + 1, device=device),
                            indexing="ij",
                        ),
                        dim=-1,
                    ),
                    start_dim=0,
                    end_dim=-2,
                )
            ],
            dim=-2,
        ).clamp(-1 + 1e-1, 1 - 1e-1)
        positions = (positions + 1) / 2  # [-1,1]->[0,1]
        indices = (positions * (self.resolution - 1)).int()
        indices = torch.cat(
            [
                indices + shift_
                for shift_ in torch.flatten(
                    torch.stack(
                        torch.meshgrid(
                            torch.arange(2, device=device),
                            torch.arange(2, device=device),
                            torch.arange(2, device=device),
                            indexing="ij",
                        ),
                        dim=-1,
                    ),
                    start_dim=0,
                    end_dim=-2,
                )
            ],
            dim=1,
        )

        indices = (
            indices[..., 0]
            + indices[..., 1] * self.resolution
            + indices[..., 2] * self.resolution**2
        )  # T x P x 3 -> T x P x 1 ; [0,res]
        indices = (
            indices
            + torch.arange(0, self.T, dtype=torch.long, device=device)[:, None]
            * self.resolution**3
        ).flatten()  # T x [0,res] -> [0,(T*res**3)*res] include time_shift in indices
        indices = torch.unique(indices).to(device)
        self.grid_mask = torch.zeros(
            self.T * self.resolution**3, dtype=torch.bool, device=device
        )
        self.grid_mask[indices] = True
        self.grid_mask = (
            self.grid_mask.reshape(
                -1, self.resolution, self.resolution, self.resolution
            )
            .permute(0, 3, 2, 1)
            .flatten()
        )
        print(
            "   Reduced Grid-Points by {}%, from [{}] to [{}] Points".format(
                round(
                    100 - (indices.shape[0] / (self.resolution**3 * self.T)) * 100, 2
                ),
                self.resolution**3 * self.T,
                indices.shape[0],
            )
        )

    def _compute_neighbours(self):
        res = self.resolution
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(res),
                torch.arange(res),
                torch.arange(res),
                indexing="ij",
            ),
            dim=-1,
        )[None, ...].repeat(self.T, 1, 1, 1, 1)
        time_shift = (
            torch.ones(self.T, res, res, res)
            * (torch.arange(0, self.T, dtype=torch.long) * res**3)[:, None, None, None]
        )

        grid_mask = self.grid_mask.reshape(-1, res, res, res).cpu()
        positions_masked = positions[grid_mask]
        time_shift_masked = time_shift[grid_mask]

        # compute neighbour index
        neighbour_x1 = positions_masked + torch.tensor([[1, 0, 0]])  # x neighbour
        neighbour_y1 = positions_masked + torch.tensor([[0, 1, 0]])  # y neighbour
        neighbour_z1 = positions_masked + torch.tensor([[0, 0, 1]])  # z neighbour
        neighbour_x2 = positions_masked + torch.tensor([[-1, 0, 0]])  # x neighbour
        neighbour_y2 = positions_masked + torch.tensor([[0, -1, 0]])  # y neighbour
        neighbour_z2 = positions_masked + torch.tensor([[0, 0, -1]])  # z neighbour

        # masks of out of bounds
        boundary_mask_x1 = (
            (neighbour_x1 < res)
            .all(dim=-1)
            .logical_and((neighbour_x1 >= 0).all(dim=-1))
        )
        boundary_mask_y1 = (
            (neighbour_y1 < res)
            .all(dim=-1)
            .logical_and((neighbour_y1 >= 0).all(dim=-1))
        )
        boundary_mask_z1 = (
            (neighbour_z1 < res)
            .all(dim=-1)
            .logical_and((neighbour_z1 >= 0).all(dim=-1))
        )
        boundary_mask_x2 = (
            (neighbour_x2 < res)
            .all(dim=-1)
            .logical_and((neighbour_x2 >= 0).all(dim=-1))
        )
        boundary_mask_y2 = (
            (neighbour_y2 < res)
            .all(dim=-1)
            .logical_and((neighbour_y2 >= 0).all(dim=-1))
        )
        boundary_mask_z2 = (
            (neighbour_z2 < res)
            .all(dim=-1)
            .logical_and((neighbour_z2 >= 0).all(dim=-1))
        )

        # 3D Index -> 1D Index
        neighbour_x1 = (
            neighbour_x1[..., 2]
            + neighbour_x1[..., 1] * res
            + neighbour_x1[..., 0] * res**2
        )
        neighbour_y1 = (
            neighbour_y1[..., 2]
            + neighbour_y1[..., 1] * res
            + neighbour_y1[..., 0] * res**2
        )
        neighbour_z1 = (
            neighbour_z1[..., 2]
            + neighbour_z1[..., 1] * res
            + neighbour_z1[..., 0] * res**2
        )
        neighbour_x2 = (
            neighbour_x2[..., 2]
            + neighbour_x2[..., 1] * res
            + neighbour_x2[..., 0] * res**2
        )
        neighbour_y2 = (
            neighbour_y2[..., 2]
            + neighbour_y2[..., 1] * res
            + neighbour_y2[..., 0] * res**2
        )
        neighbour_z2 = (
            neighbour_z2[..., 2]
            + neighbour_z2[..., 1] * res
            + neighbour_z2[..., 0] * res**2
        )
        positions_masked = (
            positions_masked[..., 2]
            + positions_masked[..., 1] * res
            + positions_masked[..., 0] * res**2
        )

        # shift in time
        positions_shifted = positions_masked + time_shift_masked
        neighbour_shifted_x1 = (neighbour_x1 + time_shift_masked)[boundary_mask_x1]
        neighbour_shifted_y1 = (neighbour_y1 + time_shift_masked)[boundary_mask_y1]
        neighbour_shifted_z1 = (neighbour_z1 + time_shift_masked)[boundary_mask_z1]
        neighbour_shifted_x2 = (neighbour_x2 + time_shift_masked)[boundary_mask_x2]
        neighbour_shifted_y2 = (neighbour_y2 + time_shift_masked)[boundary_mask_y2]
        neighbour_shifted_z2 = (neighbour_z2 + time_shift_masked)[boundary_mask_z2]

        mask_x1 = self.grid_mask[neighbour_shifted_x1.long()].cpu()
        mask_y1 = self.grid_mask[neighbour_shifted_y1.long()].cpu()
        mask_z1 = self.grid_mask[neighbour_shifted_z1.long()].cpu()
        mask_x2 = self.grid_mask[neighbour_shifted_x2.long()].cpu()
        mask_y2 = self.grid_mask[neighbour_shifted_y2.long()].cpu()
        mask_z2 = self.grid_mask[neighbour_shifted_z2.long()].cpu()

        self.adj = torch.cat(
            (
                torch.cat(
                    (
                        positions_shifted[boundary_mask_x1][mask_x1][:, None],
                        neighbour_shifted_x1[mask_x1][:, None],
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        positions_shifted[boundary_mask_y1][mask_y1][:, None],
                        neighbour_shifted_y1[mask_y1][:, None],
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        positions_shifted[boundary_mask_z1][mask_z1][:, None],
                        neighbour_shifted_z1[mask_z1][:, None],
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        positions_shifted[boundary_mask_x2][mask_x2][:, None],
                        neighbour_shifted_x2[mask_x2][:, None],
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        positions_shifted[boundary_mask_y2][mask_y2][:, None],
                        neighbour_shifted_y2[mask_y2][:, None],
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        positions_shifted[boundary_mask_z2][mask_z2][:, None],
                        neighbour_shifted_z2[mask_z2][:, None],
                    ),
                    dim=-1,
                ),
            )
        )

        self.A = self.adj.shape[-1]
        new_indices = torch.unique(self.adj).to(self.indices.device)
        pruned_indices = torch.zeros(
            self.resolution**3 * self.T, dtype=torch.long, device=self.indices.device
        )
        pruned_indices[new_indices.long()] = torch.arange(
            new_indices.shape[0], device=self.indices.device
        )
        self.adj = pruned_indices[self.adj.long()].permute(1, 0)

    def _compute_matrix(self, smoothness):
        V = self.adj.max() + 1
        idx = torch.arange(V, dtype=torch.long, device=self.indices.device)
        eye = torch.sparse_coo_tensor(
            torch.stack((idx, idx), dim=0),
            torch.ones(V, dtype=torch.float, device=self.indices.device),
            (V, V),
        )

        self.L_space = self.laplacian_uniform_points()
        self.M = (eye + self.L_space * smoothness).coalesce()

    def laplacian_uniform_points(self) -> torch.Tensor:
        adj_values = torch.ones(
            self.adj.shape[1], device=self.indices.device, dtype=torch.float
        )
        diag_idx, counts = torch.unique(self.adj, return_counts=True)

        idx = torch.cat((self.adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, counts))

        return torch.sparse_coo_tensor(
            idx, values, (self.adj.max() + 1, self.adj.max() + 1)
        ).coalesce()
