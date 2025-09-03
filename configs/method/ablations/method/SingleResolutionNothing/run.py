import os

os.system(
    "python Main.py --device 2 --target obj -np 5000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_Nothing -m configs/method/ablations/method/SingleResolutionNothing/SingleResolution.json "
)
