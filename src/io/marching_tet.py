import kaolin
import torch
import tqdm

from ext.dynoSurf.code.model.dmtet_network import Decoder
from ext.dynoSurf.code.utils.tet_utils import build_tet_grid, calc_tet_edge_length
from ext.dynoSurf.code.utils.utils import Batch_index_select

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.structures import Meshes


def train_template(args: dict, sample: dict):
    """
    Train the template mesh using the keyframe points.
    We make use of the default parameters for the template optimization, as presented by the Dynosurf Code.
    """
    device = args.device
    if sample["points"].ndim == 2:
        sample["points"] = sample["points"].unsqueeze(0)
    if sample["normals"].ndim == 2:
        sample["normals"] = sample["normals"].unsqueeze(0)
    if sample["points"].device != device:
        sample["points"] = sample["points"].to(device)
    if sample["normals"].device != device:
        sample["normals"] = sample["normals"].to(device)

    tet_verts, tets, convex_verts, convex_faces = build_tet_grid(
        sample["points"].squeeze().cpu().numpy(),
        args.io_args["out_path"],
        5.5e-7,
    )
    if not isinstance(tet_verts, torch.Tensor):
        tet_verts = torch.from_numpy(tet_verts).to(device).float()
    else:
        tet_verts = tet_verts.to(device).float()
    if not isinstance(tets, torch.Tensor):
        tets = torch.from_numpy(tets).to(device).long()
    else:
        tets = tets.to(device).long()
    if not isinstance(convex_verts, torch.Tensor):
        convex_verts = torch.from_numpy(convex_verts).to(device).float()
    else:
        convex_verts = convex_verts.to(device).float()
    if not isinstance(convex_faces, torch.Tensor):
        convex_faces = torch.from_numpy(convex_faces).to(device).long()
    else:
        convex_faces = convex_faces.to(device).long()

    edge_len_min, edge_len_mean = calc_tet_edge_length(tet_verts, tets)
    if edge_len_min > 1e-4:
        tet_edge_len = edge_len_min
    else:
        tet_edge_len = edge_len_mean

    decoder_args = {"internal_dims": 128, "hidden": 5, "multires": 5}
    shape_model = Decoder(
        convex_verts=convex_verts.cpu().numpy(),
        convex_faces=convex_faces.cpu().numpy(),
        **decoder_args
    ).to(device)

    shape_model.pre_train_convex(5000, tet_verts=tet_verts)

    vars = [p for _, p in shape_model.named_parameters()]
    optimizer = torch.optim.Adam(vars, lr=1e-4)

    # init
    for it in tqdm.tqdm(range(5000)):

        loss, mesh_verts, mesh_faces = fitting_template(
            tet_verts, tets, tet_edge_len, sample, it, shape_model
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, mesh_verts, mesh_faces = fitting_template(
        tet_verts, tets, tet_edge_len, sample, 5000, shape_model
    )
    return mesh_verts, mesh_faces


def fitting_template(
    tet_verts: torch.Tensor,
    tets: torch.Tensor,
    tet_edge_len: torch.Tensor,
    sample: torch.Tensor,
    it: int,
    shape_model: Decoder,
):

    pred = shape_model(tet_verts)
    sdf, deform = pred[:, 0], pred[:, 1:]
    verts_deform = tet_verts + torch.tanh(deform) * tet_edge_len

    mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(
        verts_deform.unsqueeze(0), tets, sdf.unsqueeze(0)
    )

    mesh_verts = mesh_verts[0]
    mesh_faces = mesh_faces[0]

    if mesh_verts.shape[0] == 0:
        print("mesh_verts.shape = ", mesh_verts.shape)
        exit()

    loss = calc_template_loss(
        mesh_verts.unsqueeze(0), mesh_faces, verts_deform, sdf, sample, it
    )
    verts_deform_diff = torch.norm(verts_deform - tet_verts)

    return loss, mesh_verts.detach(), mesh_faces.detach()


def calc_template_loss(verts_deformed, faces, tet_verts_deformed, sdf, sample, it):

    meshes = Meshes(
        verts=list([verts_deformed[i, :, :] for i in range(verts_deformed.shape[0])]),
        faces=list([faces for i in range(verts_deformed.shape[0])]),
    )

    loss = 0

    chamfer_loss, normal_error = calc_chamfer(meshes, sample)
    loss += 500 * chamfer_loss

    if normal_error is not None:
        loss += 0.001 * normal_error

    sdf_loss = approximate_sdf_loss(tet_verts_deformed.unsqueeze(0), sample, sdf)
    loss += 50 * sdf_loss

    return loss


def approximate_sdf_loss(deform_tet_verts, sample, pred_sdf, k=10, r=0.1):
    # tet_verts, bxnx3
    # bxnxm
    idx = knn_points(deform_tet_verts, sample["points"], K=k).idx
    b, n, _ = idx.shape
    idx = idx.reshape(b, n * k)

    # bxnkx3
    closest_points = Batch_index_select(sample["points"], idx).reshape(b, n, k, 3)
    closest_normals = Batch_index_select(sample["normals"], idx).reshape(b, n, k, 3)

    diff = deform_tet_verts.unsqueeze(-2) - closest_points
    # bxnxk
    pp = torch.norm(diff, dim=-1)
    weight = torch.exp(-(pp**2) / (r * r))
    sfmax = torch.nn.Softmax(dim=-1)
    weight = sfmax(weight)
    ppl = torch.sum(closest_normals * diff, dim=-1)

    # mask = sample["template_mask"].unsqueeze(-1)
    sdf = torch.sum(weight * ppl, dim=-1)
    fuse_sdf = torch.sum(sdf, dim=0)

    L1loss = torch.nn.L1Loss()
    asdf_loss = L1loss(pred_sdf, fuse_sdf)
    return asdf_loss


def calc_chamfer(meshes, sample):

    pred_points, pred_normals = sample_points_from_meshes(
        meshes, 10000, return_normals=True
    )

    chamfer, normal_error = chamfer_distance(
        x=pred_points,
        y=sample["points"],
        x_normals=pred_normals,
        y_normals=sample["normals"],
        norm=1,
    )

    return chamfer, normal_error
