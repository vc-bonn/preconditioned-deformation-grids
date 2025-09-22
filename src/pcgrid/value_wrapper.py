import torch
from argparse import Namespace
import itertools
from functools import reduce
import os

from pcgrid.utils import list_class_names, validate_dict
from pcgrid.optimization_values import *

from pcgrid.grid import MultiresolutionGrid

required_keys = {"grids"}


class ValueWrapper(torch.nn.Module):
    """
    Class to wrap different value parameterizations for optimization.
    Either it uses a grid-based approach or a default parameterization.
    This class is designed to be flexible and extensible, allowing to optimize
    different types of value transformations such as translation, rotation,
    opacity, and scales.
    Uses default parameterization for values not defined in the grid.

    It supports multiple independant grids, each with its own set of parameters.
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        validate_dict(self.wrapper_args, required_keys)
        self.input_keys = {"grid_index", "points"}

        self.wrappers = []
        for value_def in self.wrapper_args["grids"]:
            if not isinstance(self.wrapper_args["grids"][value_def], dict):
                raise ValueError("Grid must be a dict.")
            else:
                self.wrappers.append(
                    GridWrapper(self.args, self.wrapper_args["grids"][value_def])
                )
        wrapped_parameters = [
            type(p).__name__
            for p in list(
                itertools.chain.from_iterable(
                    [w.parameterizations for w in self.wrappers]
                )
            )
        ]
        assert len(wrapped_parameters) == len(
            set(wrapped_parameters)
        ), "Parameterization definition duplication"
        for parameter in list_class_names(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "optimization_values.py"
            )
        ):
            if (
                parameter not in wrapped_parameters
                and parameter in self.wrapper_args["defaults"]
            ):
                self.wrappers.append(DefaultWrapper(self.args, parameter))

    @property
    def device(self) -> torch.device:
        """Returns the device."""
        return torch.device(self.args.device)

    @property
    def wrapper_args(self) -> dict:
        """Returns the wrapper arguments."""
        return self.args["wrapper_args"]

    def zero_grad(self):
        """Zeroes the gradients of all wrappers."""
        for wrapper in self.wrappers:
            wrapper.zero_grad()

    def step(self):
        """Steps the optimization of all wrappers."""
        for wrapper in self.wrappers:
            wrapper.step()

    def forward(self, data: dict) -> torch.Tensor:
        """Combines the outputs of all wrappers into a single dictionary."""
        validate_dict(data, self.input_keys)
        assert (
            data["points"].min() >= -1 and data["points"].max() <= 1
        ), "Points must be in the range [-1, 1]."
        # assert (
        #     data["grid_index"].shape[0] == data["points"].shape[0]
        # ), "Grid index and points must have the same batch size."
        return reduce(lambda a, b: a | b, [wrapper(data) for wrapper in self.wrappers])


class SupportWrapper(torch.nn.Module):
    """
    Supports the ValueWrapper class by providing a common interface
    for different value parameterizations. It defines the basic structure
    and methods that all parameterizations should implement, such as zero_grad,
    step, and get_parameterization.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def device(self) -> torch.device:
        """Returns the device."""
        return torch.device(self.args["device"])

    @property
    def wrapper_args(self) -> dict:
        """Returns the wrapper arguments."""
        return self.args["wrapper_args"]

    def pad_parameters(self, d: dict) -> dict:
        return d | {"device": self.device}

    def zero_grad(self):
        """Zeroes the gradients of the parameterization."""
        raise NotImplementedError(
            "This method should be implemented in subclasses to zero the gradients."
        )

    def step(self):
        """Steps the optimization of the parameterization."""
        raise NotImplementedError(
            "This method should be implemented in subclasses to step the optimization."
        )

    def get_parameterization(self, key: str) -> torch.nn.Module:
        """Returns the parameterization based on the value type."""
        if key in self.wrapper_args["parameterization"]:
            parameter_args = self.pad_parameters(
                self.wrapper_args["parameterization"][key]
            )
        else:
            parameter_args = self.pad_parameters({"method": "default"})
        if key == "Translation":
            return Translation(parameter_args)
        elif key == "Rotation":
            return Rotation(parameter_args)
        elif key == "Opacity":
            return Opacity(parameter_args)
        elif key == "Scales":
            return Scales(parameter_args)
        else:
            raise ValueError(f"Unsupported value type: {key}")


class GridWrapper(SupportWrapper):
    """
    Wrapper for grid-based parameterizations.
    It initializes the grid and its parameterizations based on the provided grid definition.
    It also provides methods to step the optimization and zero the gradients of the grid.
    """

    def __init__(self, args: Namespace, grid_def: dict):
        super().__init__(args)
        self.parameterizations = [
            self.get_parameterization(value) for value in grid_def["parameters"]
        ]
        n_grid_values = sum([v.grid_values for v in self.parameterizations])
        grid_args = self.pad_dict(grid_def["grid_args"], n_grid_values)

        self.grid = MultiresolutionGrid(grid_args)

    def zero_grad(self):
        return self.grid.zero_grad()

    def step(self):
        """Steps the optimization of the grid."""
        self.grid.step()

    def pad_dict(self, grid_args: dict, n_values: int) -> dict:
        """Pads the grid arguments to insert the missing values"""
        padded_args = grid_args.copy()
        padded_args["grid_values"] = n_values
        padded_args["device"] = self.device
        return padded_args

    def forward(self, data: dict) -> dict:
        """Returns the values of the parameterizations for the given data by sampling the preconditioned grid."""
        values = self.grid(data)
        grid_values = 0
        output = {}
        for parameterization in self.parameterizations:
            output[type(parameterization).__name__] = parameterization(
                values[..., grid_values : grid_values + parameterization.grid_values]
            )
            grid_values += parameterization.grid_values
        return output


class DefaultWrapper(SupportWrapper):
    """A wrapper for the Default values of a parameterization."""

    def __init__(self, args: Namespace, parameterization: str):
        super().__init__(args)
        self.parameterization = self.get_parameterization(parameterization)

    def forward(self, data: dict) -> dict:
        values = torch.ones(
            data["grid_index"].shape[0], data["points"].shape[-2], device=self.device
        )
        return {type(self.parameterization).__name__: self.parameterization(values)}

    def zero_grad(self):
        """No gradients to zero for default parameterizations."""
        pass

    def step(self):
        """No optimization step needed for default parameterizations."""
        pass
