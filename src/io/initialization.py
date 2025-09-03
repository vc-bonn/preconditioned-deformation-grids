import sys
import torch
import pymeshlab


def scale_points(points: torch.Tensor):
    """
    scales Points to fit in range [0, 1]
    """
    max_ps = points.max(0).values
    min_ps = points.min(0).values
    mean_ps = points.mean(0)
    bounding_box_len = (max_ps - min_ps).norm()

    points2 = (points - mean_ps) / bounding_box_len

    return points2, bounding_box_len, mean_ps


def revert_scale_points(
    scaled_points: torch.Tensor, bounding_box_len: torch.Tensor, mean_ps: torch.Tensor
):
    """
    reverts the scaling of points
    """
    return scaled_points * bounding_box_len + mean_ps


def poisson(points: torch.Tensor, normals: torch.Tensor):
    """
    Perform Poisson surface reconstruction on the given points and normals.
    """
    if points.shape[0] == 0:
        raise ValueError("No points provided for Poisson reconstruction.")
    if normals.shape[0] == 0:
        raise ValueError("No normals provided for Poisson reconstruction.")
    if points.shape[0] != normals.shape[0]:
        raise ValueError("Points and normals must have the same number of elements.")
    if points.shape[1] != 3 or normals.shape[1] != 3:
        raise ValueError("Points and normals must be 3D vectors.")
    if points.dtype != torch.float32 or normals.dtype != torch.float32:
        raise ValueError("Points and normals must be of type float32.")
    if points.dim() != 2 or normals.dim() != 2:
        raise ValueError("Points and normals must be 2D tensors.")
    v = points.to(torch.float64).cpu().squeeze().numpy()
    n = normals.cpu()
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=v, v_normals_matrix=n)
    ms.add_mesh(m, "mesh")
    ms.generate_surface_reconstruction_screened_poisson(
        threads=64, pointweight=10, depth=10
    )
    verts = torch.Tensor(ms.current_mesh().vertex_matrix()).to(torch.float32)
    faces = torch.Tensor(ms.current_mesh().face_matrix()).to(torch.int64)
    return verts, faces


def marching_tetras(args: dict, points: torch.Tensor, normals: torch.Tensor):
    """
    Perform marching tetrahedra on the keyframe points to create a mesh.
    """
    from src.io.marching_tet import train_template

    points, min_vals, max_vals = scale_points(points)
    verts, faces = train_template(
        sample={"points": points, "normals": normals}, args=args
    )
    verts = revert_scale_points(verts, min_vals, max_vals)
    return verts, faces


def diffusion(args: dict, points: torch.Tensor, normals: torch.Tensor):

    sys.path.append("ext/m2v/core/network")
    sys.path.append("ext/m2v")
    import ext.m2v.core.network.models_ae as models_ae
    import ext.m2v.core.network.models_diff as models_diff
    from ext.m2v.im2mesh.utils.onet_generator import get_generator

    points = (points) / 2

    shape_ae = models_ae.__dict__["kl_d512_m512_l8"]()
    print("Loading shape ae %s" % "ckpts/DT4D/vae/dt4d_shape_ae.pth")
    shape_ae.load_state_dict(
        torch.load("ext/m2v/ckpts/DT4D/vae/dt4d_shape_ae.pth", map_location="cpu")[
            "model"
        ]
    )
    shape_ae.to(args.device)
    shape_ae.eval()

    shape_model = models_diff.__dict__["kl_d512_m512_l8_surf512_edm"]()
    print(
        "Loading shape dm %s" % "ext/m2v/ckpts/DT4D/sparse/dt4d_shape_diff_sparse.pth"
    )
    shape_model.load_state_dict(
        torch.load(
            "ext/m2v/ckpts/DT4D/sparse/dt4d_shape_diff_sparse.pth", map_location="cpu"
        )["model"]
    )
    shape_model.to(args.device)
    shape_model.eval()

    shape_sampled_array = shape_model.sample(
        args.device, random=True, cond=points[None]
    ).float()

    generator = get_generator()

    first_frame = generator.generate_from_latent(
        z=shape_sampled_array[:1], F=shape_ae.decode
    )
    verts = first_frame.vertices
    faces = first_frame.faces

    return torch.Tensor(verts).to(torch.float32) * 2, torch.Tensor(faces).to(
        torch.int64
    )
