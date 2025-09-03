from torch.utils.data.dataset import Dataset
import torch


class OptimizationDataset(Dataset):
    def __init__(self, args: dict, data: dict) -> None:  # T x P x 3
        super().__init__()
        self.args = args
        self.data = data
        self.points = data["points"].to(self.device)
        self.normals = data["normals"].to(self.device)

    @property
    def device(self):
        return self.args.device

    @property
    def io_args(self):
        return self.args.io_args

    def __len__(self) -> int:
        return self.points.shape[0] - 1

    def __getitem__(self, index) -> dict:
        if index < self.args.method_args["keyframe_index"]:
            return {
                "points": self.points[index + 1],
                "normals": self.normals[index],
                "index": torch.tensor([index + 1], device=self.device),
                "target": self.points[index],
                "target_index": torch.tensor([index], device=self.device),
                "grid_index": torch.tensor([index], device=self.device),
            }
        else:
            return {
                "points": self.points[index],
                "normals": self.normals[index + 1],
                "index": torch.tensor([index], device=self.device),
                "target": self.points[index + 1],
                "target_index": torch.tensor([index + 1], device=self.device),
                "grid_index": torch.tensor([index], device=self.device),
            }
