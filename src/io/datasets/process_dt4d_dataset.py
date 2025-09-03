import numpy as np

import os
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
import torch

if __name__ == "__main__":

    ama_path = "/data/kaltheuner/dt4d_v3"
    out_path = "/data/kaltheuner/processed_data/DT4D"
    os.makedirs(out_path, exist_ok=True)

    lengths = [20]

    directories = [
        os.path.join(ama_path, d)
        for d in os.listdir(ama_path)
        if os.path.isdir(os.path.join(ama_path, d))
    ]
    for dir_path in directories:
        obj = dir_path.split("/")[-1]
        dir_path = os.path.join(dir_path, "static_1")
        obj_files = [
            int(f.split(".")[0]) for f in os.listdir(dir_path) if f[-4:] == ".npz"
        ]
        obj_files.sort()
        obj_files = [str(o) + ".npz" for o in obj_files]
        for l in lengths:
            pos_idx = 0
            # while (len(obj_files) - pos_idx) // l > 0:
            seq_path = os.path.join(out_path, str(l), obj + "_" + str(pos_idx))
            os.makedirs(seq_path, exist_ok=True)
            seq_files = obj_files[pos_idx : pos_idx + l]
            points = []
            for f in seq_files:
                for x in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                    data = np.load(os.path.join(dir_path[:-1] + x, f))
                    points.append(
                        torch.from_numpy(data["canonical_view_pc"]).to(torch.float32),
                    )
                # points.append()
                pcl = Pointclouds(
                    [torch.cat(points)],
                )
                pcl = pcl.subsample(5000)
                pcl.estimate_normals()
                IO().save_pointcloud(
                    pcl, os.path.join(seq_path, f.split(".")[0] + ".ply")
                )
                # shutil.copyfile(os.path.join(dir_path, f), os.path.join(seq_path, f))
            pos_idx += l
