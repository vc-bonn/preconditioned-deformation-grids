from pcgrid.value_wrapper import ValueWrapper
import torch

# Example arguments for ValueWrapper
args = {
    "device": "cuda:0",  # define the device to use
    "wrapper_args": {  # define all arguments for the wrapper
        "parameterization": {  # define the parameterizations to use via preconditioned grids
            "Translation": {  # Class name of the parameterization (see src/optimization_values.py)
                "grid_values": 3,  # Number of values at each grid cell
                "method": "tanh",  # How to transform the grid outputs (see src/optimization_values.py)
            },
        },
        "grids": {  # Dict containing all grids to use
            "grid_0": {  # Each grid definition supports unique hyperparameters
                "parameters": ["Translation"],  # Which parameters to bind to this grid
                "grid_args": {  # Hyperparameters for the grid
                    "base_res": 1,  # Resolution of the grid at the coarsest level
                    "increase_per_level": 4,  # How much to increase the resolution at each level
                    "n_level": 2,  # Number of levels in the grid
                    "lr_adapt": 1.2,  # Learning rate adaption factor forch each grid level
                    "local_lambda_adapt": 1.5,  # Local smoothness lambda adaption factor for each grid level
                    "smoothness": 0.25,  # Smoothness of the grid at the coarsest level
                    "lr": 0.1,  # Learning rate for the grid at the coarsest level
                    "solver": "CG",  # Solver to use for the grid (see src/grid/solver.py)
                    "T": 2,  # Number of time steps for the grid,
                    "T_lambda_dampening": 0,  # Dampening factor for the smoothness for the time connections
                },
            }
        },
        "defaults": [
            "Rotation"
        ],  # Default parameterizations to use if not specified in the grid
    },
}

if __name__ == "__main__":
    wrapper = ValueWrapper(args)
    print("Wrapper initialized with args:", wrapper.wrapper_args, "\n")

    print("One Set of points for a single timestep:")
    # Point Shape
    points = (
        torch.rand(1, 100, 3, device=args["device"]) * 2 - 1
    )  # Example points in 3D space
    print("input points shape: ", points.shape, "dtype: ", points.dtype)
    print("min: ", points.min(), "max: ", points.max())

    data = {
        "grid_index": torch.zeros(1, device=args["device"], dtype=torch.long),
        "points": points[:, None, None, :, :],
    }
    print(
        "grid index shape: ", data["grid_index"].shape, "values: ", data["grid_index"]
    )
    values = wrapper(data)
    print("Output values shape: ", {k: v.shape for k, v in values.items()}, "\n\n")

    print("Two Set of points for a single timestep:")
    points = (
        torch.rand(2, 100, 3, device=args["device"]) * 2 - 1
    )  # Example points in 3D space
    print("input points shape: ", points.shape)

    data = {
        "grid_index": torch.zeros(2, device=args["device"], dtype=torch.long),
        "points": points[:, None, None, :, :],
    }
    print(
        "grid index shape: ", data["grid_index"].shape, "values: ", data["grid_index"]
    )
    values = wrapper(data)
    print("Output values shape: ", {k: v.shape for k, v in values.items()}, "\n\n")

    print("Two Set of points for two timestep:")
    points = (
        torch.rand(2, 100, 3, device=args["device"]) * 2 - 1
    )  # Example points in 3D space
    print("input points shape: ", points.shape)

    data = {
        "grid_index": torch.arange(2, device=args["device"], dtype=torch.long),
        "points": points[:, None, None, :, :],
    }
    print(
        "grid index shape: ", data["grid_index"].shape, "values: ", data["grid_index"]
    )
    values = wrapper(data)
    print("Output values shape: ", {k: v.shape for k, v in values.items()}, "\n \n")

    print("Example of optimization: 2 Grid Levels")
    points = (
        torch.rand(1, 100, 3, device=args["device"]) * 2 - 1
    )  # Example points in 3D space
    target = torch.zeros(1, 100, 3, device=args["device"])
    print("Target positions shape: ", target.shape, "values: ", target[0, 0])

    data = {
        "grid_index": torch.zeros(1, device=args["device"], dtype=torch.long),
        "points": points[:, None, None, :, :],
    }

    for ep in range(1000):
        wrapper.zero_grad()
        output = wrapper(data)

        positions = output["Translation"] + data["points"].squeeze(dim=(1, 2))
        loss = torch.nn.functional.mse_loss(positions, target)
        if ep % 250 == 0:
            print(f"Epoch {ep}, Loss: {round(loss.item(),6)}")
        loss.backward()
        wrapper.step()
    print(f"Epoch {ep}, Loss: {round(loss.item(),6)} \n \n")

    print("Example of optimization: 4 Grid Levels")
    args["wrapper_args"]["grids"]["grid_0"]["grid_args"]["n_level"] = 4
    wrapper = ValueWrapper(args)
    points = (
        torch.rand(1, 100, 3, device=args["device"]) * 2 - 1
    )  # Example points in 3D space
    target = torch.zeros(1, 100, 3, device=args["device"])
    print("Target positions shape: ", target.shape, "values: ", target[0, 0])

    data = {
        "grid_index": torch.zeros(1, device=args["device"], dtype=torch.long),
        "points": points[:, None, None, :, :],
    }

    for ep in range(1000):
        wrapper.zero_grad()
        output = wrapper(data)

        positions = output["Translation"] + data["points"].squeeze(dim=(1, 2))
        loss = torch.nn.functional.mse_loss(positions, target)
        if ep % 250 == 0:
            print(f"Epoch {ep}, Loss: {round(loss.item(),6)}")
        loss.backward()
        wrapper.step()
    print(f"Epoch {ep}, Loss: {round(loss.item(),6)}")
