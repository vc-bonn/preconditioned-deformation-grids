import os
import tqdm
import torch
from torch.utils.data import DataLoader
from src.io.datasets.existing import existingDataset
from src.optimization import Optimization

from src.utilities.util import (
    generate_output_path,
    scale_points,
    initialize_meshes,
    revert_scale,
)


from pcgrid.value_wrapper import ValueWrapper

from src.io.log import Log
from src.geometry.geometry import Geometry
import json


class Opt_Run:
    def __init__(self, args: dict):
        self.args = args
        self._run_directory()

    @property
    def method_args(self):
        return self.args.method_args

    @property
    def io_args(self):
        return self.args.io_args

    def _run_directory(self) -> None:
        if self.io_args["directory_path"][-1] != "/":
            self.io_args["directory_path"] = self.io_args["directory_path"] + "/"
        directories = os.listdir(self.io_args["directory_path"])
        bar = tqdm.tqdm(directories)
        for i, directory in enumerate(bar):
            if "skip" in self.io_args.keys():
                if i < self.io_args["skip"] or not os.path.isdir(
                    self.io_args["directory_path"] + directory + "/"
                ):
                    continue
            print("\n#####\n", directory, "\n#####\n")
            self.io_args["input_directory"] = (
                self.io_args["directory_path"] + directory + "/"
            )
            self.io_args["directory"] = directory + "_" + self.method_args["descriptor"]
            self.input_path = (
                self.io_args["directory_path"] + self.io_args["directory"] + "/"
            )
            generate_output_path(self.args)

            self.input_dataset = existingDataset(self.args)
            dataloader = DataLoader(
                self.input_dataset,
                batch_size=self.input_dataset.__len__(),
                num_workers=0,
            )

            for _, data in enumerate(dataloader):
                for key in data.keys():
                    if isinstance(data[key], dict):
                        for subkey in data[key].keys():
                            if isinstance(data[key][subkey], torch.Tensor):
                                if data[key][subkey].shape[0] == 1:
                                    data[key][subkey] = data[key][subkey][0]
                                if data[key][subkey].dtype == torch.float64:
                                    data[key][subkey] = data[key][subkey].to(
                                        torch.float32
                                    )
                    elif isinstance(data[key], torch.Tensor):
                        if data[key].shape[0] == 1:
                            data[key] = data[key][0]
                        if data[key].dtype == torch.float64:
                            data[key] = data[key].to(torch.float32)

                self(data)

    def compute_keyframe(self, points, res=512):
        if self.args.keyframe == "ours":
            positions = (points[..., :3].squeeze() + 1) / 2
            positions = (positions * res).int()
            outputs = torch.stack(
                [
                    torch.tensor([(torch.unique(p_, dim=0) / res).shape[0]])
                    for p_ in positions
                ]
            )
            x = torch.exp(
                -0.001
                * torch.arange(-points.shape[0] // 2, points.shape[0] // 2, 1) ** 2
            )[:, None]
            index = (outputs * x).argmax().item()
            print("Init Index: [{}]".format(index))

            # set min/max indices for opt_dataset due to edgecase when index = 0 or index = max
            self.args.t_min = 0 if index > 0 else 1
            self.args.t_max = (
                points.shape[0] - 1
                if index < points.shape[0] - 1
                else points.shape[0] - 2
            )
            return index
        elif self.args.keyframe == "first":
            self.args.t_min = 1
            self.args.t_max = points.shape[0] - 1
            return 0
        elif self.args.keyframe == "middle":
            self.args.t_min = 0
            self.args.t_max = points.shape[0] - 1
            return points.shape[0] // 2
        else:
            raise Exception("Unknown Keyframe Method [{}]".format(self.args.keyframe))

    def init_surf(self, points, normals):
        if self.args.init == "ours":
            from src.io.initialization import poisson

            return poisson(points, normals)

        elif self.args.init == "tetra":
            from src.io.initialization import marching_tetras

            return marching_tetras(self.args, points.squeeze()[..., :3], normals)

        elif self.args.init == "diffusion":
            from src.io.initialization import diffusion

            return diffusion(self.args, points.squeeze()[..., :3], normals)

        else:
            raise Exception("Unknown Init Method [{}]".format(self.args.init))

    def __call__(self, data: dict) -> None:
        self.args.T = data["points"].shape[0] - 1

        #####
        # Preprocess Target Points
        #####
        print("Number Target Points per T: [{}]".format(data["points"].shape[-2]))
        target_points, self.args.points_min, self.args.points_max = scale_points(
            data["points"]
        )
        target_points = target_points.to(self.args.device)

        #####
        # Keyframe Selection
        #####
        self.method_args["keyframe_index"] = self.compute_keyframe(
            target_points.squeeze(), res=self.args.init_grid_resolution
        )

        #####
        # Initial Mesh
        #####
        vert, face = self.init_surf(
            target_points[self.method_args["keyframe_index"]].squeeze()[:, :3],
            data["normals"][self.method_args["keyframe_index"]],
        )
        verts, faces = initialize_meshes(self.args, vert, face)

        #####
        # Create Log | Optimizer | Grid | Mesh Optimizer Classes
        #####
        opt = Optimization(self.args)
        opt_mesh = Geometry(
            self.method_args["geometry"],
            verts.to(self.args.device),
            faces.to(self.args.device),
        )
        wrapper_args = {
            "device": self.args.device,  # define the device to use
            "wrapper_args": {  # define all arguments for the wrapper
                "parameterization": {  # define the parameterizations to use via preconditioned grids
                    "Translation": {  # Class name of the parameterization (see src/optimization_values.py)
                        "grid_values": 3,  # Number of values at each grid cell
                        "method": "tanh",  # How to transform the grid outputs (see src/optimization_values.py)
                    },
                    "Rotation": {"grid_values": 3, "method": "cayley"},
                },
                "grids": {  # Dict containing all grids to use
                    "grid_0": {  # Each grid definition supports unique hyperparameters
                        "parameters": [
                            "Rotation",
                            "Translation",
                        ],  # Which parameters to bind to this grid
                        "grid_args": self.method_args["grid"]
                        | {
                            "T": self.args.T,
                            "T_lambda_dampening": -1,
                        },  # Grid hyperparameters
                    }
                },
                "defaults": [],  # Default parameterizations to use if not specified in the grid
            },
        }
        opt_grid = ValueWrapper(wrapper_args)

        #####
        # Run
        #####
        data["points"] = target_points
        meshes, losses = opt(opt_grid, opt_mesh, data)

        #####
        # Metrics | Output
        #####
        eval_dict = revert_scale(
            data,
            meshes,
            self.args.points_min,
            self.args.points_max,
            self.io_args["out_path"],
        )
        self.log_metrics(eval_dict)

    @torch.no_grad()
    def log_metrics(self, data: dict):
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
