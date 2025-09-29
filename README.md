<<<<<<< HEAD

# Preconditioned Deformation Grids
![](assets/teaser.jpg)

## Install
```
git clone https://github.com/vc-bonn/preconditioned-deformation-grids.git pdg
cd pdg

git submodule update --init

conda create -n T python=3.12
conda activate T


pip install ext/pcgrid
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install tqdm tensorboard scikit-learn charonload cmake gpytoolbox imageio matplotlib ninja open3d opencv-python pykdtree trimesh pymeshlab openmesh tensorboardx
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Main.py Arguments
```
python Main.py --...
[-m / --methocConfig] - Path to the method config file (str)
[-se / --seed] - random seed (int)
[-d / --device] - cuda device number (int)
[-t / --target] - which target type to use. If pcl is chosen, make sure that the directory path is as configured in the AMA dataset. ("pcl"/"obj")
[-np / --number_points] - if [-t == "obj"] set the number of target points sampled per obj file (int)
[-o / --out_path] - output directory (str)
[-dp / --directory_path] - path to the input objects, see PATH-STRUCTURE section (str)
[-s / --skip] - how many objects to skip (int)
```

## PATH-STRUCTURE
To download the preprocessed data, see the data section of [Dynosurf](https://github.com/yaoyx689/DynoSurf?tab=readme-ov-file).
``` 
--AMA
|
|--crane_0010
|-|
|-|--gt
|-|--pcl_seqs
|-|--points_clouds
|
|--crane_0027
|-|
|-|--gt
|-|--pcl_seqs
|-|--points_clouds
```

## RUNS
All run configs are predefined in configs/method/runs. For the ablation study see the configs/method/ablations path. Alter the path arguments accordingly [-o / -dp]
``` 
[AMA] python configs/method/runs/run_ama.py
[DT4D] python configs/method/runs/run_dt4d.py
[DFAUST] python configs/method/runs/run_dfaust.py
```

To run the sequence length ablation studies, use the scripts in src/io/dataset/process_[___]_dataset.py to prepare the input data. These scripts have been tested and demonstrated on the AMA dataset. 
Please note that the code has been refactored, so results may differ slightly from those reported in the paper. 
The noise ablation study is currently not re-enabled.
To run the initialization methods from DynoSurf or Motion2VecSets, make sure to install their required packages. For Motion2VecSets, we recommend creating a separate conda environment with the appropriate Python version to avoid version conflicts.

=======
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
git clone https://github.com/vc-bonn/pcgrid.git
pip install pcgrid/
```

### Usage See examples in example.py:
main class in `src/value_wrapper.py` is `ValueWrapper`, which can be initialized with a dictionary of arguments.
```python
from pcgrid.value_wrapper import ValueWrapper
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
The output of the `ValueWrapper` is a dictionary containing the computed values for each parameter. The keys are the names of the parameterization classes, and the values are tensors with the output values.
>>>>>>> 7297116272283b84b992c7d3da6c1a9069ce2352
