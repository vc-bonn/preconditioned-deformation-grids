import torch

"""
We define several value wrappers for different types of transformations
such as translation, rotation, opacity, and scales. Each wrapper inherits from
the base `Value` class and implements a forward method to create the respective
transformation matrices or values.

tranlation, rotation, opacity, and scales are common transformations
used in computer graphics, gaussian splatting, registration, and other
applications.
"""


class Value(torch.nn.Module):
    """
    Base class for value parameterizations.
    """

    def __init__(self, args: dict):
        super().__init__()
        self.args = args

    @property
    def device(self) -> torch.device:
        """Returns the device."""
        return torch.device(self.args["device"])

    @property
    def grid_values(self) -> int:
        """Returns the number of grid values."""
        return self.args["grid_values"]

    def forward():
        """Forward pass to create values."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class Translation(Value):
    """A wrapper for translation values."""

    def __init__(self, args: dict):
        super().__init__(args)
        if self.args["method"] not in ["tanh", "default"]:
            raise ValueError(f"Unsupported translation method: {self.args['method']}")
        self.translation = torch.zeros(3, device=self.device, dtype=torch.float32)[
            None, None, :
        ]

    def forward(self, values) -> torch.Tensor:
        """Forward pass to create translation matrices."""
        if self.args["method"] == "tanh":
            return torch.tanh(values)
        elif self.args["method"] == "default":
            return self.translation.repeat(values.shape[0], values.shape[1], 1)


class Rotation(Value):
    """A wrapper for rotation values."""

    def __init__(self, args: dict):
        super().__init__(args)
        if self.args["method"] not in ["cayley", "default"]:
            raise ValueError(f"Unsupported rotation method: {self.args['method']}")
        if self.args["method"] == "default":
            self.rotation = torch.eye(4, device=self.device, dtype=torch.float32)[
                None, None, :, :
            ]  # Identity matrix for default parameterization

    def cayley(self, rotation_params: torch.Tensor) -> torch.Tensor:
        """Converts rotation parameters to a rotation matrix using Cayley transform."""
        assert rotation_params.shape[-1] == 3, "Rotation parameters must have 6 values."
        T = rotation_params.reshape(-1, 3)
        R = torch.zeros(T.shape[0], 4, 4, device=self.device)
        R[..., 0, 1] = T[:, 2]
        R[..., 0, 2] = T[:, 1]
        R[..., 1, 2] = -T[:, 0]
        R[..., 1, 0] = T[:, 2]
        R[..., 2, 0] = -T[:, 1]
        R[..., 2, 1] = T[:, 0]
        eye = torch.eye(4, device=self.device)[None]
        R = ((eye + R).inverse() @ (eye - R)) ** 2
        return R.reshape(rotation_params.shape[0], -1, 4, 4)

    def forward(self, values) -> torch.Tensor:
        """Forward pass to create rotation matrices."""
        if self.args["method"] == "cayley":
            return self.cayley(values)
        elif self.args["method"] == "default":
            return self.rotation.repeat(values.shape[0], values.shape[1], 1, 1)


class Opacity(Value):
    """A wrapper for opacity values."""

    def __init__(self, args: dict):
        super().__init__(args)
        if self.args["method"] not in ["default", "direct"]:
            raise ValueError(f"Unsupported opacity method: {self.args['method']}")
        self.opacity = torch.zeros(1, device=self.device, dtype=torch.float32)[
            None, :
        ]  # Default opacity

    def forward(self, values) -> torch.Tensor:
        """Forward pass to create opacity values."""
        if self.args["method"] == "default":
            return self.opacity.expand(values.shape[0], values.shape[1])
        elif self.args["method"] == "direct":
            return values.squeeze(dim=-1)  # Directly use the provided values
        else:
            raise ValueError(f"Unsupported opacity method: {self.args['method']}")


class Scales(Value):
    """A wrapper for scale values."""

    def __init__(self, args: dict):
        super().__init__(args)
        if self.args["method"] not in ["default", "direct"]:
            raise ValueError(f"Unsupported scale method: {self.args['method']}")
        self.scale = torch.zeros(3, device=self.device, dtype=torch.float32)[
            None, None, :
        ]  # Default scale

    def forward(self, values) -> torch.Tensor:
        """Forward pass to create scale values."""
        if self.args["method"] == "default":
            return self.scale.expand(values.shape[0], values.shape[1], -1)
        elif self.args["method"] == "direct":
            return values  # Directly use the provided values
        else:
            raise ValueError(f"Unsupported scale method: {self.args['method']}")
