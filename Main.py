import argparse

parser = argparse.ArgumentParser(description="nvdiffrecmc")
parser.add_argument(
    "-m",
    "--methodConfig",
    type=str,
    default="configs/method/fit.json",
    help="Method config file",
)
parser.add_argument("-se", "--seed", type=int, default=0)
parser.add_argument("-d", "--device", type=str, default="0")
parser.add_argument("-t", "--target", type=str, default="obj")
parser.add_argument("-np", "--number_points", type=int, default=5000)
parser.add_argument("-irs", "--init_grid_resolution", type=int, default=128)
parser.add_argument("-o", "--out_path", type=str, default="test")
parser.add_argument("-i", "--init", type=str, default="ours")
parser.add_argument("-k", "--keyframe", type=str, default="ours")
parser.add_argument(
    "-dp",
    "--directory_path",
    type=str,
    default="/data/kaltheuner/DT4D",
)
parser.add_argument("-s", "--skip", type=int, default=0)


args = parser.parse_args()

import torch
import numpy as np
import random
import json
import os

from src.opt_run import Opt_Run

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


if __name__ == "__main__":
    args.device = "cuda:" + args.device
    torch.multiprocessing.set_start_method("spawn")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.io_args = {
        "base_out_path": args.out_path,
        "skip": args.skip,
        "directory_path": args.directory_path,
        "noise": 0,
    }

    if os.path.isfile(args.methodConfig):
        with open(args.methodConfig) as json_file:
            args.method_args = json.load(json_file)
            if "edgeloss" in args.method_args["optimization"].keys():
                print(
                    "Regularization Loss Enabled with weight [{}]".format(
                        args.method_args["optimization"]["edgeloss"]
                    )
                )
                args.io_args["base_out_path"] += "_Regularized"

                args.io_args["base_out_path"] += "_init[{}]".format(args.init)

                args.io_args["base_out_path"] += "_keyframe[{}]".format(args.keyframe)
    else:
        raise Exception("Method Config File is not a File [Path not Found]")

    if args.io_args["noise"] > 0.0:
        raise Exception("Noise Ablation Study is currently not re-enabled.")
    Opt_Run(args)
