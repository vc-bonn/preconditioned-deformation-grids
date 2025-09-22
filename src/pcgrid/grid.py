import torch
from largesteps.parameterize import to_differential
from typing import Tuple
from pcgrid.utils import validate_dict
from pcgrid.solver import from_differential

required_keys = {
    "smoothness",
    "local_lambda_adapt",
    "n_level",
    "base_res",
    "increase_per_level",
    "lr",
    "lr_adapt",
    "device",
    "T",
    "solver",
    "T_lambda_dampening",
}


class MultiresolutionGrid(torch.nn.Module):
    def __init__(self, grid_args: dict, verbose: bool = False):
        super().__init__()
        validate_dict(grid_args, required_keys)

        self.grids = []
        self.grid_args = grid_args
        if grid_args is None:
            print("No grid arguments provided, using default values.")

        args = self.parameters_per_level()
        for arg in args:
            if verbose:
                print("Grid {}".format(arg))

            if self.grid_args["T_lambda_dampening"] > 0:
                self.grids.append(Grid_level(grid_args=arg))
            else:
                self.grids.append(Split_Grid_Level(grid_args=arg))
        self.zero_grad()

    def zero_grad(self):
        for grid in self.grids:
            grid.zero_grad()

    def step(self):
        """Steps the optimization of the grids."""
        for grid in self.grids:
            grid.step()

    def parameters_per_level(self) -> Tuple[list, list, list]:
        smoothness = [
            self.grid_args["smoothness"] * self.grid_args["local_lambda_adapt"] ** i
            for i in range(self.grid_args["n_level"])
        ]
        resolutions = [
            self.grid_args["base_res"] + self.grid_args["increase_per_level"] * i
            for i in range(self.grid_args["n_level"])
        ]
        lrs = [
            self.grid_args["lr"] * self.grid_args["lr_adapt"] ** i
            for i in range(self.grid_args["n_level"])
        ]
        args = [
            {key: value for key, value in self.grid_args.items()}
            for _ in range(self.grid_args["n_level"])
        ]
        for arg, sm, res, lr in zip(args, smoothness, resolutions, lrs):
            arg["smoothness"] = sm
            arg["resolution"] = res
            arg["lr"] = lr
        return args

    def forward(self, data: dict):
        if len(data) > 0:
            output = torch.stack([grid.forward(data) for grid in self.grids])
            return output.mean(dim=0)
        else:
            return 0


class Split_Grid_Level(torch.nn.Module):
    def __init__(self, grid_args: dict):
        super().__init__()
        T = grid_args["T"]
        grid_args["T"] = 1
        self.grids = [Grid_level(grid_args=grid_args) for _ in range(T)]
        pass

    def forward(self, data: dict) -> torch.Tensor:
        values = []
        for grid_idx in data["grid_index"]:
            # if grid_idx == 0:
            #     d = {"points": data["points"].detach(), "grid_index": 0}
            # else:
            d = {"points": data["points"], "grid_index": 0}

            values.append(self.grids[grid_idx](d))
        return torch.cat(values, dim=0)

    def zero_grad(self):
        for grid in self.grids:
            grid.zero_grad()

    def step(self):
        for grid in self.grids:
            grid.optimizer.step()


class Grid_level(torch.nn.Module):
    def __init__(self, grid_args: dict = None):
        super().__init__()
        self.grid_args = grid_args
        opt_parameters = torch.zeros(
            (self.T, self.grid_size, self.grid_size, self.grid_size, self.cell_values),
            dtype=torch.float32,
            device=self.device,
        ).flatten(start_dim=0, end_dim=-2)

        if self.grid_size > 1 or self.T > 1:
            self.precondition()
            self.opt_values = to_differential(self.M, opt_parameters)
        else:
            self.opt_values = opt_parameters
        self.opt_values.requires_grad_()

        self.optimizer = torch.optim.Adam(params=[self.opt_values], lr=self.lr)

    @property
    def grid_size(self):
        return self.grid_args["resolution"]

    @property
    def T(self):
        return self.grid_args["T"]

    @property
    def device(self):
        return self.grid_args["device"]

    @property
    def smoothness(self):
        return self.grid_args["smoothness"]

    @property
    def cell_values(self):
        return self.grid_args["grid_values"]

    @property
    def lr(self):
        return self.grid_args["lr"]

    def precondition(self):
        """
        define the neighbourhood and compute the Laplacian matrix and create the M matrix
        """

        L_time, L_space = None, None
        if self.grid_size > 1:
            adjacencies_space = self.define_neighbourhood_space()
            L_space = self.laplacian_uniform_points(adjacencies_space)
        if self.T > 1:
            adjacencies_time = self.define_neighbourhood_time()
            L_time = self.laplacian_uniform_points(adjacencies_time)

        if L_space is not None and L_time is not None:
            V = (
                (torch.cat((adjacencies_space, adjacencies_time), dim=-1) + 1)
                .max()
                .to(torch.int32)
            )
        elif L_space is not None:
            V = (adjacencies_space + 1).max().to(torch.int32)
        elif L_time is not None:
            V = (adjacencies_time + 1).max().to(torch.int32)
        else:
            raise ValueError("Grid size and T cannot both be 1.")
        idx = torch.arange(V, dtype=torch.int32, device=self.device)
        eye = torch.sparse_coo_tensor(
            torch.stack((idx, idx), dim=0),
            torch.ones(V, dtype=torch.float, device=self.device),
            (V, V),
        )

        if L_space is not None:
            self.M = eye + L_space * self.smoothness
        elif L_time is not None:
            self.M = (
                eye + L_time * self.smoothness * self.grid_args["T_lambda_dampening"]
            )

        if L_time is not None and L_space is not None:
            self.M += L_time * self.smoothness
        self.M = self.M.coalesce()

    def laplacian_uniform_points(self, adjacencies: torch.Tensor) -> torch.Tensor:
        """
        Compute the Laplacian matrix for uniform points.
        """
        if adjacencies.shape[-1] == 0:
            return None
        adj_values = torch.ones(
            adjacencies.shape[1], device=self.device, dtype=torch.float
        )
        V = (adjacencies.max() + 1).to(torch.int32)
        diag_idx, counts = torch.unique(adjacencies, return_counts=True)

        idx = torch.cat((adjacencies, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, counts))

        return torch.sparse_coo_tensor(idx.to(torch.int32), values, (V, V)).coalesce()

    def define_neighbourhood_space(self):
        """
        Define the neighbourhood of the grid points in 3D space and return the adjacency matrix containing the parameter indices.
        """
        # Define the positions of the grid points and their neighbours
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(self.grid_size, device=self.device),
                torch.arange(self.grid_size, device=self.device),
                torch.arange(self.grid_size, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).flatten(
            start_dim=0, end_dim=-2
        )  # N, 3

        neighbours = torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            device=self.device,
        )  # 6, 3
        neighbours_indices = positions[:, None, :] + neighbours[None, :, :]  # N, 6, 3

        # Ensure indices are within bounds
        in_bounds = ((neighbours_indices >= 0).all(dim=-1)).logical_and(
            (neighbours_indices < self.grid_size).all(dim=-1)
        )  # N, 6

        # 3d index to 1d index
        positions = (
            positions[..., 0] * self.grid_size * self.grid_size
            + positions[..., 1] * self.grid_size
            + positions[..., 2]
        )  # N**3
        neighbours_indices = (
            neighbours_indices[..., 0] * self.grid_size * self.grid_size
            + neighbours_indices[..., 1] * self.grid_size
            + neighbours_indices[..., 2]
        )  # N**3, 6

        # Add time dimension
        time = torch.arange(self.T, device=self.device) * self.grid_size**3  # T
        positions = positions[None, :] + time[:, None]  # T, N**3
        neighbours_indices = (
            neighbours_indices[None, :] + time[:, None, None]
        )  # T, N**3, 6
        in_bounds = in_bounds[None, :, :].repeat(self.T, 1, 1)  # T, N**3, 6

        # create adjacency matrix
        neighbours_indices = neighbours_indices[in_bounds].flatten()
        positions = (
            positions[:, :, None].repeat(1, 1, in_bounds.shape[-1])[in_bounds].flatten()
        )
        return torch.stack([positions, neighbours_indices], dim=0)

    def define_neighbourhood_time(self):
        """
        Define the neighbourhood of the grid points in 3D space and return the adjacency matrix containing the parameter indices.
        """
        # Define the positions of the grid points and their neighbours
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(self.grid_size, device=self.device),
                torch.arange(self.grid_size, device=self.device),
                torch.arange(self.grid_size, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).flatten(
            start_dim=0, end_dim=-2
        )  # N, 3

        neighbours = torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            device=self.device,
        )  # 6, 3
        neighbours_indices = positions[:, None, :] + neighbours[None, :, :]  # N, 6, 3

        # 3d index to 1d index
        positions = (
            positions[..., 0] * self.grid_size * self.grid_size
            + positions[..., 1] * self.grid_size
            + positions[..., 2]
        )  # N**3
        neighbours_indices = (
            neighbours_indices[..., 0] * self.grid_size * self.grid_size
            + neighbours_indices[..., 1] * self.grid_size
            + neighbours_indices[..., 2]
        )  # N**3, 6

        # Add time dimension
        time = torch.arange(self.T, device=self.device) * self.grid_size**3  # T
        positions = positions[None, :] + time[:, None]  # T, N**3

        # Add time Neighbours
        neighbours_time_indices = torch.tensor(
            [-self.grid_size**3, self.grid_size**3], device=self.device
        )  # 2

        neighbours_time = (
            positions[:, :, None] + neighbours_time_indices[None, None, :]
        )  # T, N**3, 2

        in_bounds_time = (neighbours_time >= 0).logical_and(
            neighbours_time < self.grid_size**3 * self.T
        )  # T, N**3, 2

        # create adjacency matrix
        neighbours_indices = neighbours_time[in_bounds_time].flatten()
        positions = (
            positions[:, :, None]
            .repeat(1, 1, in_bounds_time.shape[-1])[in_bounds_time]
            .flatten()
        )
        return torch.stack([positions, neighbours_indices], dim=0)

    def forward(self, data: dict) -> torch.Tensor:
        if hasattr(self, "M"):
            opt_values = from_differential(
                self.M, self.opt_values, self.T, method=self.grid_args["solver"]
            )
        else:
            opt_values = self.opt_values
        grid = opt_values.reshape(
            self.T, self.grid_size, self.grid_size, self.grid_size, self.cell_values
        )[data["grid_index"]]
        if grid.ndim == 4:
            grid = grid[None, ...]

        values = torch.nn.functional.grid_sample(
            grid.permute(0, 4, 1, 2, 3),
            data["points"],
            padding_mode="border",
            align_corners=True,
        ).squeeze(dim=(-3, -2))

        # Debug
        # pos = (((data["points"] + 1) / 2) * self.grid_size).long()
        # pos = pos[0].flatten(end_dim=-2).unique(dim=0)
        # print("Unique positions:", pos.shape)
        # print(
        #     "Grid Coverage: {:.2f}%".format(
        #         (pos.shape[0] * 100)
        #         / (self.grid_size * self.grid_size * self.grid_size)
        #     )
        # )
        return values.permute(0, 2, 1)  # T, P, C

    def zero_grad(self):
        self.opt_values.grad = None
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
