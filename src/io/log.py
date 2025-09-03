import torch
from tensorboardX import SummaryWriter
import json
import os


class Log:
    def __init__(self, args) -> None:
        self.args = args
        self.writer = SummaryWriter(log_dir=self.io_args["out_path"])

    @property
    def io_args(self):
        return self.args.io_args

    def log_sequence(
        self, sequence: torch.Tensor, identifier: str, epoch=int , fps=3
    ):
        if sequence.ndim == 4:
            sequence = sequence[None]
        self.writer.add_video(
            identifier,
            sequence[..., :3].permute(0, 1, 4, 2, 3),
            global_step=epoch,
            fps=fps,
        )
        return

    def log_meshes(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        identifier: str,
        idxs: list,
        epoch: int = None,
        color: torch.Tensor = None,
    ):
        vertices = verts[idxs]
        if vertices.ndim == 2:
            vertices = vertices[None]
        if color is not None:
            color = color[idxs].clamp(0, 1)
        if faces is not None:
            assert faces.ndim == 2
            faces = faces[None].repeat(len(idxs), 1, 1)
        self.writer.add_mesh(
            "{}_{}".format(identifier, idxs),
            vertices,
            faces=faces,
            colors=color,
            global_step=epoch,
        )

    @torch.no_grad()
    def log_scalars(
        self,
        value: torch.Tensor,
        identifier: str,
        epoch: int = None,
        reduction="mean",
    ):
        if value.ndim == 0:
            self.writer.add_scalar(
                identifier,
                value,
                epoch,
            )
        else:
            if reduction == "sum":
                value = value.sum()
            elif reduction == "mean":
                value = value.mean()
            self.writer.add_scalar(
                identifier,
                value,
                epoch,
            )

    @torch.no_grad()
    def log_metrics(self, data: dict, losses: dict):
        self.writer.add_scalar("eval/chamfer_distance", data["chamfer_distance"])
        self.writer.add_scalar("eval/normals", data["normals"])
        self.writer.add_scalar("eval/f05", data["f05"])
        self.writer.add_scalar("eval/f1", data["f1"])
        with open(
            os.path.join(
                self.io_args["out_path"],
                "method_args.json",
            ),
            "w",
        ) as json_file:
            json.dump(
                self.args.method_args,
                json_file,
            )
        with open(
            os.path.join(
                self.io_args["out_path"],
                "Eval.json",
            ),
            "w",
        ) as json_file:
            json.dump(
                data,
                json_file,
            )
