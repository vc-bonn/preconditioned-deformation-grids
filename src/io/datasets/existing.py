import torch
import openmesh as om
import os

from pytorch3d.io import load_objs_as_meshes, save_ply, load_ply
from pytorch3d.ops import sample_points_from_meshes, estimate_pointcloud_normals
import numpy as np
from torch.utils.data.dataset import Dataset


class existingDataset(Dataset):
    def __init__(self, args: dict):
        self.args = args
        if args.target == "ply":
            pcl_files = [
                int(f.split("/")[-1].split(".")[0])
                for f in self._get_files(
                    os.path.join(self.io_args["input_directory"], "point_clouds"),
                    ".ply",
                )
            ]
            pcl_files.sort()
            pcl_files = [str(format(file, "04")) + ".ply" for file in pcl_files]
            meshes = [
                load_ply(
                    os.path.join(
                        os.path.join(self.io_args["input_directory"], "point_clouds"),
                        file,
                    )
                )
                for file in pcl_files
            ]

            self.points = torch.stack([mesh[0].to(torch.float32) for mesh in meshes])
            self.normals = estimate_pointcloud_normals(
                self.points, neighborhood_size=16
            )
            pass
        elif args.target == "obj":
            obj_files = [
                int(f.split("/")[-1].split(".")[0])
                for f in self._get_files(
                    os.path.join(self.io_args["input_directory"], "gt"),
                    ".obj",
                )
            ]
            obj_files.sort()
            obj_files = [str(format(file, "04")) + ".obj" for file in obj_files]
            meshes = load_objs_as_meshes(
                [
                    os.path.join(
                        os.path.join(self.io_args["input_directory"], "gt"),
                        file,
                    )
                    for file in obj_files
                ]
            )

            self.points, self.normals = sample_points_from_meshes(
                meshes, self.args.number_points, return_normals=True
            )
        else:
            raise NotImplementedError(
                "Target type {} not implemented".format(args.target)
            )
        obj_files = self._get_files(
            self.io_args["input_directory"],
            ".obj",
        )
        if len(obj_files) == 0:
            obj_files = self._get_files(
                os.path.join(self.io_args["input_directory"], "gt"), ".obj"
            )
        obj_files.sort()
        meshes = load_objs_as_meshes(obj_files)

        self.gt_points = meshes.verts_padded()
        self.gt_normals = meshes.verts_normals_padded()
        self.gt_faces = meshes.faces_padded()

        os.makedirs(
            os.path.join(self.io_args["out_path"], "input_meshes"), exist_ok=True
        )
        for i, (points, normals) in enumerate(zip(self.points, self.normals)):
            save_ply(
                os.path.join(self.io_args["out_path"], "input_meshes", "%04d.ply" % i),
                verts=points,
                verts_normals=normals,
                faces=None,
            )
        #####
        # TODO: Reimplement noise addition
        #####
        # if "noise" in self.io_args.keys():
        #     from pytorch3d.structures import Meshes, Pointclouds

        #     mesh = Pointclouds(points=[target_points[..., :3].reshape(-1, 3)])
        #     bb = mesh.get_bounding_boxes().squeeze()
        #     d = (bb[:, 0] - bb[:, 1]).norm()
        #     noise = (
        #         d * torch.randn(target_points[..., :3].shape) * self.io_args["noise"]
        #     )
        #     target_points[..., :3] += noise
        #     for i, points in enumerate(target_points):
        #         os.makedirs(
        #             os.path.join(self.io_args["out_path"], "input_meshes_noisy"),
        #             exist_ok=True,
        #         )
        #         save_ply(
        #             os.path.join(
        #                 self.io_args["out_path"], "input_meshes_noisy", "%04d.ply" % i
        #             ),
        #             verts=points.squeeze()[:, :3],
        #             faces=None,
        #         )

    @property
    def io_args(self):
        return self.args.io_args

    def _get_files(self, file_path: str, type: str) -> str:
        if file_path[-1] != "/":
            file_path += "/"
        files = [
            file_path + i
            for i in os.listdir(file_path)
            if os.path.isfile(os.path.join(file_path, i)) and type in i
        ]
        return files

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {
            "points": self.points[index],
            "normals": self.normals[index],
            "gt_points": self.gt_points[index],
            "gt_normals": self.gt_normals[index],
            "gt_faces": self.gt_faces[index],
        }
