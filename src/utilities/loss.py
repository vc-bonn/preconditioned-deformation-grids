from src.utilities.util import temp_sys_path

with temp_sys_path("ext/dynoSurf/code"):
    from loss import Loss
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import torch
from pytorch3d.structures import Meshes


# Wrapper Class for DynoSurf Loss Code
class Loss_f(Loss):
    def __init__(self, args) -> None:
        self.args = args

        params = {
            "TemplateOptimize": {
                "normal_weight": None,
                "chamfer_weight": None,
                "sdf_weight": None,
            },
            "JointOptimize": {
                "chamfer_weight": None,
                "sdf_sign_weight": None,
                "normal_weight": None,
                "deform_smooth_weight": None,
                "template_weight": None,
                "sdf_sigmoid_scale": None,
                "sdf_fusion_thres": None,
                "robust_chamfer_param": None,
            },
            "Training": {"sampling_num_pts": 100000},
        }
        super().__init__(params)

    def calc_chamfer(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        target_points: torch.Tensor,
        target_normals: torch.Tensor,
        l=1,
        single_direction=False,
    ):
        if faces is not None:
            meshes = Meshes(verts, faces)
            pred_points, pred_normals = sample_points_from_meshes(
                meshes, self.sampling_num, return_normals=True
            )
        else:
            pred_points = verts
            pred_normals = None

        chamfer_error, normal_error = chamfer_distance(
            x=target_points,
            y=pred_points,
            x_normals=target_normals,
            y_normals=pred_normals,
            norm=l,
            batch_reduction=None,
            single_directional=single_direction,
        )
        return chamfer_error, normal_error
