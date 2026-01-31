import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.loss import chamfer_distance

from gpytoolbox import remesh_botsch
import open3d as o3d
import numpy as np
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from contextlib import contextmanager
import sys


@contextmanager
def temp_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.pop(0)


def generate_output_path(args):
    path = os.path.join(args.io_args["base_out_path"], args.io_args["directory"])
    for i in range(2200):
        if not os.path.isdir(path + "/" + str(i)):
            args.io_args["out_path"] = path + "/" + str(i)
            mkdirs(args.io_args["out_path"])
            return
    raise Exception(
        "Too Many Output Folders | Clean Up The Output Directory [", path, "]"
    )


def mkdirs(path: str) -> None:
    mkdir_path = ""
    if path[0] == "/":
        mkdir_path += "/"
    for directory in path.split("/"):
        if directory == "":
            continue
        mkdir_path += directory + "/"
        if not os.path.isdir(mkdir_path):
            os.mkdir(mkdir_path)


def revert_scale(input, meshes, points_min, points_max, out_path):
    def re_scale(points):
        points = points.cpu().squeeze()[..., :3]
        points /= 0.95
        points *= (points_max.cpu() - points_min.cpu()).max() / 2
        points += (points_max.cpu() + points_min.cpu()) / 2
        return points

    points = re_scale(meshes.verts_padded()).cpu()
    input_points = input["gt_points"].cpu()

    for idx, (p, f) in enumerate(zip(points, meshes.faces_padded())):
        save_obj(
            os.path.join(
                out_path,
                "%04d.obj" % idx,
            ),
            p,
            f,
        )

    from ext.dynosurf.evaluation.utils import eval_pointcloud

    cd_list = []
    nc_list = []
    f1_list = []
    f05_list = []
    for p_, g_, p_f, g_f in zip(
        points.cpu(), input_points.cpu(), meshes.faces_padded(), input["gt_faces"]
    ):
        pred_mesh = trimesh.Trimesh(vertices=p_.cpu().numpy(), faces=p_f.cpu().numpy())
        goal_mesh = trimesh.Trimesh(vertices=g_.cpu().numpy(), faces=g_f.cpu().numpy())
        pred_sample, idx = pred_mesh.sample(100000, return_index=True)
        pred_normals = pred_mesh.face_normals[idx]
        goal_sample, idx = goal_mesh.sample(100000, return_index=True)
        goal_normals = goal_mesh.face_normals[idx]
        result_dict = eval_pointcloud(
            pred_sample, goal_sample, pred_normals, goal_normals
        )
        cd_list.append(result_dict["chamfer-L2"])
        nc_list.append(result_dict["normals"])
        f1_list.append(result_dict["f-score"])
        f05_list.append(result_dict["f-score-5"])

    return {
        "chamfer_distance": np.array(cd_list).mean(),
        "normals": np.array(nc_list).mean(),
        "f05": np.array(f05_list).mean(),
        "f1": np.array(f1_list).mean(),
    }


def scale_points(points):
    points = points.squeeze()[..., :3]
    points_min = (points.min(dim=0)[0]).min(dim=0)[0]
    points_max = (points.max(dim=0)[0]).max(dim=0)[0]
    points = scale(points, points_min, points_max)
    points = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    return points[:, None, None, :, :], points_min, points_max


def scale(points: torch.Tensor, points_min, points_max):
    points -= (points_max + points_min) / 2
    points /= (points_max - points_min).max() / 2
    points *= 0.95
    return points


def initialize_meshes(args, verts, faces):
    save_obj(
        os.path.join(
            args.io_args["out_path"],
            "init_geometry_{}.obj".format(verts.shape[0]),
        ),
        verts,
        faces,
    )

    verts = verts.cpu().double().numpy()
    faces = faces.cpu().to(torch.int32).numpy()

    while True:
        verts, faces = remesh_botsch(
            verts,
            faces,
            5,
            None,
            True,
        )
        print("Init Mesh: [{} x 3]".format(verts.shape[0]))
        print("-----")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
        mesh.compute_vertex_normals()

        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=min(20000, args.number_points * 4)
        )
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

        mesh_1 = Meshes(
            torch.tensor(v).to(torch.float32)[None],
            torch.tensor(f).to(torch.int32)[None],
        )
        mesh_2 = Meshes(
            torch.tensor(verts).to(torch.float32)[None],
            torch.tensor(faces).to(torch.int32)[None],
        )
        p_1 = sample_points_from_meshes(mesh_1, num_samples=100000)
        p_2 = sample_points_from_meshes(mesh_2, num_samples=100000)

        distance, _ = chamfer_distance(p_1.to(args.device), p_2.to(args.device))
        if distance < 5e-4:
            break
        else:
            print(
                "Mesh Simplification Failed [Open3D error], repeating mesh initialization"
            )

    verts, faces = remesh_botsch(
        v,
        f,
        5,
        None,
        True,
    )
    verts = torch.from_numpy(v).to(torch.float32).to(args.device)
    faces = torch.from_numpy(f).to(torch.int64).to(args.device)

    save_obj(
        os.path.join(
            args.io_args["out_path"],
            "init_geometry_{}.obj".format(verts.shape[0]),
        ),
        verts,
        faces,
    )
    print("Geometry: [{} x 3]".format(verts.shape[0]))
    return verts, faces


def edgelength(v: torch.Tensor, f: torch.Tensor):

    mesh = Meshes(
        verts=v[..., :3].squeeze()[None],
        faces=f[None],
    )
    edges_packed = mesh.edges_packed()
    verts_packed = mesh.verts_packed()
    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    return (v0 - v1).norm(dim=-1, p=2)
