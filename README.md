# Preconditioned Grids
This repository contains implementations of preconditioned grids for various optimization tasks.  
The code is designed to be modular and extensible, allowing for easy integration of new parameterizations.  
Exemplary parameterizations include translation, rotation, opacity, and scales, usefull for tasks like image registration, 3D reconstruction, and gaussian splatting.

It supports smoothing in space and time. In space, neighboring grid cells are connected via a smoothness term, which is controlled by a lambda parameter. Concerning the time smoothing, the grid cells are connected in time via the space smoothness term, which is controlled by a time dampening factor. If no time smoothing is desired, set the time dampening factor to 0 and T to 1.


### Required Packages - will be automatically installed if not already present:
[PyTorch](https://pytorch.org/) - Tested version 2.4.1  
[Largesteps](https://github.com/JuKalt/large-steps-pytorch-multiGPU.git) - Use fork to support multi-GPU training and remove unnecessary dependencies.  
[Cholespy](https://github.com/JuKalt/cholespy_multiGPU.git) - Use fork to support multi-GPU training and fix install script.  

### Installation
```bash
git clone https://github.com/JuKalt/preconditioned_grids.git
pip install preconditioned_grids/
```

### Usage See examples in example.py:
main class in `src/value_wrapper.py` is `ValueWrapper`, which can be initialized with a dictionary of arguments.
```python
from src.value_wrapper import ValueWrapper
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
                    "increase_per_level": 2,  # How much to increase the resolution at each level
                    "n_level": 10,  # Number of levels in the grid
                    "lr_adapt": 1.1,  # Learning rate adaption factor forch each grid level
                    "local_lambda_adapt": 1.5,  # Local smoothness lambda adaption factor for each grid level
                    "smoothness": 0.25,  # Smoothness of the grid at the coarsest level
                    "lr": 0.005,  # Learning rate for the grid at the coarsest level
                    "solver": "CG",  # Solver to use for the grid (see src/grid/solver.py)
                    "T": 2,  # Number of time steps for the grid,
                    "T_lambda_dampening": 0.25,  # Dampening factor for the smoothness for the time connections
                },
            }
        },
        "defaults": ["Rotation"], # Class names of the Default parameterization 
    },
}

# Initialize the wrapper with the arguments
wrapper = ValueWrapper(args) 

# Example data to pass to the wrapper
# This data should match the expected input format of the wrapper. see example.py for more details.
data = {
        "grid_index": torch.zeros(1, device=args["device"], dtype=torch.long),
        "points": points[:, None, None, :, :],
    }

# call the wrapper with the data
values = wrapper(data)
```

#### Input
The input to the `ValueWrapper` should be a dictionary containing:
- `grid_index`: A tensor of shape `(batch_size)` indicating which grid to use for the current input, with value range [0,T].
- `points`: A tensor of shape `(batch_size, num_points, 3)` containing the points for which the values are computed, with value range [-1,1].

#### Parameterizations
The parameterizations are defined in `src/optimization_values.py`.
Each parameterization class should inherit from `BaseParameterization` and implement the required methods.

Currently implemented parameterizations:
- `Translation`: For translation tasks, outputs 3D translations. Shape of output is `(batch_size, num_points, 3)`.
- `Rotation`: For rotation tasks, outputs 3D rotation matrices. Shape of output is `(batch_size, num_points, 4, 4)`.
- `Opacity`: For opacity tasks, outputs a single value per point. Shape of output is `(batch_size, num_points, 1)`.
- `Scales`: For 3D scale tasks, outputs 3D values per point. Shape of output is `(batch_size, num_points, 3)`.

#### Output
The output of the `ValueWrapper` is a dictionary containing the computed values for each parameterization. The keys are the names of the parameterizations, and the values are tensors with the computed values.
