import torch
import tqdm
from src.utilities.loss_f import Loss_f
from torch.utils.data import DataLoader

from pcgrid.value_wrapper import ValueWrapper
from src.utilities.util import edgelength

from pytorch3d.structures import Meshes
from src.io.datasets import optimization_dataset
from src.geometry.geometry import Geometry


def limit(data: dict, T: int) -> dict:
    limit_ = (data["target_index"] >= 0).logical_and(data["target_index"] <= (T - 1))
    data = {key: data[key][limit_] for key in data}
    for key in data:
        if data[key].ndim == 2 and key != "edgelength":
            data[key] = data[key][None]
    return data


def initialize_grid_prediction(
    index: int, device: str, v: torch.Tensor, edgelength: torch.Tensor
):
    data_ = {
        "initial_points": torch.cat(
            (v[None, None, None], v[None, None, None]), dim=0
        ).detach(),
        "points": torch.cat((v[None, None, None], v[None, None, None]), dim=0).detach(),
        "index": torch.tensor([index, index], device=device),
        "target_index": torch.tensor([index - 1, index], device=device),
        "transforms": torch.zeros(2, v.shape[0], 4, 4, device=device),
        "offset": torch.tensor([-1, +1], device=device),
        "scaling": torch.tensor([1, 1], device=device),
        "edgelength": torch.stack((edgelength, edgelength), dim=0),
    }
    data_["transforms"][:, :, 0, 0] = 1
    data_["transforms"][:, :, 1, 1] = 1
    data_["transforms"][:, :, 2, 2] = 1
    data_["transforms"][:, :, 3, 3] = 1
    data_["points"] = torch.cat(
        (
            data_["points"],
            torch.ones_like(data_["points"][..., :1]),
        ),
        dim=-1,
    )
    data_["initial_points"] = torch.cat(
        (
            data_["initial_points"],
            torch.ones_like(data_["initial_points"][..., :1]),
        ),
        dim=-1,
    )
    return data_


class Optimization(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_f = Loss_f(self.args)

    @property
    def method_args(self):
        return self.args.method_args

    @property
    def epochs(self):
        return self.method_args["optimization"]["epochs"]

    @property
    def idx(self):
        return self.method_args["keyframe_index"]

    def chamfer_distance(self, pred_points: torch.Tensor, goal: torch.Tensor):
        cd_grid_robust_1, _ = self.loss_f.calc_robust_chamfer_single_direction(
            pred_points,
            None,
            goal,
            None,
            return_normals=False,
            alpha=0.3,
        )
        cd_grid_robust_2, _ = self.loss_f.calc_robust_chamfer_single_direction(
            goal,
            None,
            pred_points,
            None,
            return_normals=False,
            alpha=0.3,
        )
        return cd_grid_robust_1 + cd_grid_robust_2

    def transform_points(self, points: torch.Tensor, transforms: list):
        points_ = (
            transforms["Rotation"] @ points.squeeze(dim=(1, 2))[..., None]
        ).squeeze(dim=-1)
        points_[..., :3] = points_[..., :3] + transforms["Translation"] * 0.1
        return points_

    def forward_input(self, data: dict, grid: ValueWrapper):
        if data["grid_index"].ndim == 2:
            data["grid_index"] = data["grid_index"].squeeze(dim=-1)
        input = {"points": data["points"][..., :3], "grid_index": data["grid_index"]}
        values = grid(input)
        points = self.transform_points(data["points"], values)
        return self.chamfer_distance(
            points[..., :3], data["target"][..., :3].squeeze(dim=(1, 2))
        )

    def forward_prediction(
        self, grid: ValueWrapper, v: torch.Tensor, f: torch.Tensor, data: dict, e
    ):
        edgeloss = 0
        edge_length = edgelength(v, f).detach()
        data_ = initialize_grid_prediction(self.idx, self.args.device, v, edge_length)
        cd_prediction, batch = 0, 0
        scaling = torch.zeros(self.args.T, device=self.args.device)  #
        output_points = [v]
        while (data_["target_index"] <= self.args.t_max - 1).any() or (
            data_["target_index"] >= self.args.t_min
        ).any():
            data_ = limit(data_, self.args.T)
            if data_["offset"].shape[0] == 0:
                break
            if data["grid_index"].ndim == 2:
                data["grid_index"] = data["grid_index"].squeeze(dim=-1)
            input = {
                "points": data_["points"][..., :3],
                "grid_index": data_["target_index"],
            }
            values = grid(input)
            transforms = values["Rotation"]
            transforms[..., :3, 3] = (
                transforms[..., :3, 3] + values["Translation"] * 0.1
            )
            data_["transforms"] = data_["transforms"] @ transforms
            data_["points"] = (
                data_["transforms"]
                @ data_["initial_points"].squeeze(dim=(1, 2))[..., None]
            ).squeeze(dim=-1)[:, None, None, :, :]
            # data_["points"] = self.transform_points(data_["points"], values)[
            #     :, None, None, :, :
            # ]

            if "edgeloss" in self.method_args["optimization"].keys():
                l_ = torch.stack(
                    [
                        edgelength(v, f)
                        for v in data_["points"][..., :3].squeeze(dim=(1, 2))
                    ],
                    dim=0,
                )
                edgeloss = edgeloss + torch.nn.functional.l1_loss(
                    l_, data_["edgelength"]
                )
                data_["edgelength"] = l_

            cd_prediction_ = self.chamfer_distance(
                data_["points"][..., :3].squeeze(dim=(1, 2)),
                data["target"][data_["target_index"], ..., :3].squeeze(dim=(1, 2)),
            )
            cd_prediction = (
                cd_prediction + (cd_prediction_ * data_["scaling"] ** e).mean()
            )
            scaling[data_["target_index"]] = data_["scaling"] ** e
            data_["scaling"] = data_["scaling"] * (
                1 / (1 + cd_prediction.detach() - self.min_cd).clamp(min=1)
            )

            if data_["target_index"][-1] >= self.method_args["keyframe_index"]:
                output_points.append(data_["points"][-1, ..., :3].squeeze(dim=(1, 2)))
            if (
                data_["target_index"].shape[0] == 2
                or data_["target_index"][-1] <= self.method_args["keyframe_index"]
            ):
                output_points.insert(0, data_["points"][0, ..., :3].squeeze(dim=(1, 2)))
            batch += 1
            data_["index"] = data_["index"] + data_["offset"]
            data_["target_index"] = data_["target_index"] + data_["offset"]

        return cd_prediction / batch, edgeloss / batch, output_points

    def forward(
        self, grid: ValueWrapper, opt_tran: Geometry, data: dict
    ) -> ValueWrapper:
        dataset = optimization_dataset.OptimizationDataset(self.args, data)
        dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False)

        bar = tqdm.tqdm(range(self.epochs))

        for epoch in bar:
            for data in dataloader:
                v, f = opt_tran()
                cd_mesh = self.chamfer_distance(
                    v[None, ...], dataset.points[self.idx, ..., :3].squeeze()[None]
                )
                cd_mesh.backward()

                # lazy warmup
                cd_input = self.forward_input(data, grid)
                if epoch > 100:
                    self.min_cd = cd_input.max().detach()

                    e = 1 - (((epoch) / (self.epochs)) ** 0.5)
                    cd_prediction, edgeloss, output = self.forward_prediction(
                        grid, v, f, data, e
                    )
                    edgeloss = edgeloss * self.method_args["optimization"]["edgeloss"]
                    (cd_input.mean() + cd_prediction + edgeloss).backward()
                else:
                    cd_prediction = 0
                    (cd_input).mean().backward()
                self.set_bar_description(bar, cd_mesh, cd_input, cd_prediction)

                grid.step()
                opt_tran.step()
                grid.zero_grad()
                opt_tran.zero_grad()

        meshes = Meshes(
            verts=[p.detach().cpu().squeeze() for p in output],
            faces=[f.cpu() for _ in output],
        )
        return meshes, {
            "grid_chamfer_distance": cd_prediction.mean().detach().cpu().item(),
            "mesh_chamfer_distance": cd_mesh.detach().cpu().item(),
        }

    def set_bar_description(
        self,
        bar: tqdm.tqdm,
        chamfer_distance_mesh: torch.Tensor,
        chamfer_distance_input: torch.Tensor,
        chamfer_distance_grid: torch.Tensor,
    ) -> None:
        if isinstance(chamfer_distance_mesh, torch.Tensor):
            chamfer_distance_mesh = chamfer_distance_mesh.mean()
        if isinstance(chamfer_distance_input, torch.Tensor):
            chamfer_distance_input = chamfer_distance_input.mean()
        if isinstance(chamfer_distance_grid, torch.Tensor):
            chamfer_distance_grid = chamfer_distance_grid.mean()
        value = chamfer_distance_mesh + chamfer_distance_input + chamfer_distance_grid
        bar.set_description(
            str(
                round(
                    value.detach().cpu().item(),
                    4,
                )
            )
        )
